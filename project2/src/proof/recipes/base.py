"""User-defined distillation methods.

A recipe is ordinary Python written against RecipeContext primitives —
sample / logprobs / judge / train / emit. The platform owns everything after
the recipe returns: the frozen-holdout proof gate, BLOCKED semantics, alias
promotion, fallback, rollback. Recipes cannot touch those.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol

from proof.backends.base import Backend, TrainedModel
from proof.resources import Dataset, Record


class RecipeContext:
    def __init__(self, backend: Backend, teacher: str, student_base: str,
                 emit: Callable[[dict], None]):
        self.backend = backend
        self.teacher = teacher
        self.student_base = student_base
        self._emit = emit
        self.spend_usd = 0.0

    async def sample(self, model: str, prompts: list[str]) -> list[dict | None]:
        self.spend_usd += self.backend.cost_per_call(model) * len(prompts)
        return await self.backend.sample(model, prompts)

    async def logprobs(self, model: str, prompt: str, completion: dict) -> float:
        return await self.backend.logprobs(model, prompt, completion)

    async def judge(self, rubric: str, items: list[dict]) -> list[bool]:
        self.spend_usd += self.backend.cost_per_call(self.teacher) * len(items)
        return await self.backend.judge(rubric, items)

    async def train(self, records: list[Record], config: dict[str, Any] | None = None) -> TrainedModel:
        payload = [r.model_dump() for r in records]
        model = await self.backend.train(self.student_base, payload, config or {})
        self.spend_usd += model.cost_usd
        return model

    def emit(self, event: str, **data: Any) -> None:
        self._emit({"event": event, **data})


class Recipe(Protocol):
    """Implement run(); receive the TRAIN split only. The frozen holdout is
    withheld by the platform and used exclusively by the proof gate."""

    name: str

    async def run(self, ctx: RecipeContext, train: list[Record]) -> TrainedModel: ...


_REGISTRY: dict[str, Recipe] = {}


def register(recipe: Recipe) -> Recipe:
    _REGISTRY[recipe.name] = recipe
    return recipe


def get_recipe(name: str) -> Recipe:
    if name not in _REGISTRY:
        raise KeyError(f"unknown recipe '{name}'; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered() -> list[str]:
    return sorted(_REGISTRY)
