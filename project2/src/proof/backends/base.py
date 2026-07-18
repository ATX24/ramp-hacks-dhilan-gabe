"""Execution backend contract. Recipes never talk to a provider directly —
they go through these primitives, so a recipe written today against the mock
or Bedrock backend runs unchanged on a future SageMaker QLoRA backend.
"""
from __future__ import annotations

from typing import Any, Protocol


class TrainedModel:
    def __init__(self, ref: str, base: str, cost_usd: float):
        self.ref = ref
        self.base = base
        self.cost_usd = cost_usd


class RecipeIncompatible(Exception):
    """Raised loudly when a recipe needs a capability this backend cannot
    provide (e.g. logit-level KD against a black-box API teacher)."""


class Backend(Protocol):
    name: str

    async def sample(self, model: str, prompts: list[str]) -> list[dict | None]:
        """Structured completions for the task. `model` may be a base model id
        or a TrainedModel.ref."""
        ...

    async def logprobs(self, model: str, prompt: str, completion: dict) -> float:
        ...

    async def judge(self, rubric: str, items: list[dict]) -> list[bool]:
        ...

    async def train(self, student_base: str, records: list[dict], config: dict[str, Any]) -> TrainedModel:
        ...

    def cost_per_call(self, model: str) -> float:
        ...
