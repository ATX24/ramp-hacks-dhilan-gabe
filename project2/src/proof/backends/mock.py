"""Deterministic local backend: an executable oracle plays the teacher, a
noisy version of it plays students. Lets the full loop — recipes, training,
proof gate, promotion, fallback — run and test with zero AWS calls.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from proof.backends.base import Backend, RecipeIncompatible, TrainedModel
from proof.metrics import CATEGORIES

_KNOWN = {
    "AWS": ("Amazon Web Services", "infrastructure", True, []),
    "GITHUB": ("GitHub", "software", True, []),
    "UBER": ("Uber", "travel", False, []),
    "DOORDASH": ("DoorDash", "meals", False, ["meal_over_policy"]),
    "FIGMA": ("Figma", "software", True, []),
    "DELTA": ("Delta Air Lines", "travel", False, ["requires_receipt"]),
    "OPENAI": ("OpenAI", "software", True, []),
    "WEWORK": ("WeWork", "office", False, []),
    "GOOGLE ADS": ("Google Ads", "advertising", False, []),
    "STRIPE": ("Stripe", "professional_services", False, []),
}

COSTS = {"teacher": 0.0080, "student": 0.0004}


def oracle(descriptor: str) -> dict:
    d = descriptor.upper()
    for key, (merchant, cat, sub, flags) in _KNOWN.items():
        if key in d:
            return {"merchant": merchant, "category": cat,
                    "is_subscription": sub, "policy_flags": list(flags)}
    cleaned = re.sub(r"[*#0-9]+", " ", descriptor).strip().title() or "Unknown"
    return {"merchant": cleaned, "category": "other",
            "is_subscription": False, "policy_flags": []}


def _det_rng(*parts: str) -> random.Random:
    seed = hashlib.sha256("|".join(parts).encode()).digest()
    return random.Random(int.from_bytes(seed[:8], "big"))


class MockBackend:
    """Teacher answers via the oracle. A trained student's quality rises with
    training-set size, so recipes that curate better data measurably win."""

    name = "mock"

    def __init__(self) -> None:
        self._students: dict[str, float] = {}  # ref -> error rate

    def _error_rate(self, model: str) -> float:
        if model in self._students:
            return self._students[model]
        if "teacher" in model or "pro" in model:
            return 0.0
        return 0.55  # untrained base student is bad

    async def sample(self, model: str, prompts: list[str]) -> list[dict | None]:
        err = self._error_rate(model)
        out: list[dict | None] = []
        for p in prompts:
            rng = _det_rng(model, p)
            gold = oracle(p)
            if rng.random() >= err:
                out.append(gold)
            elif rng.random() < 0.3:
                out.append(None)  # malformed / schema-invalid response
            else:
                wrong = dict(gold)
                wrong["category"] = rng.choice(sorted(CATEGORIES - {gold["category"]}))
                out.append(wrong)
        return out

    async def logprobs(self, model: str, prompt: str, completion: dict) -> float:
        raise RecipeIncompatible(
            f"backend '{self.name}' teacher is API-black-box; token logprobs unavailable"
        )

    async def judge(self, rubric: str, items: list[dict]) -> list[bool]:
        # Judge = "does the label match the oracle" with slight noise.
        verdicts = []
        for it in items:
            rng = _det_rng(rubric, it["input"])
            correct = it.get("output") == oracle(it["input"])
            verdicts.append(correct if rng.random() > 0.02 else not correct)
        return verdicts

    async def train(self, student_base: str, records: list[dict], config: dict[str, Any]) -> TrainedModel:
        clean = sum(1 for r in records if r.get("output") == oracle(r["input"]))
        n = max(1, len(records))
        # Error falls with volume and label cleanliness; floors at 2%.
        err = max(0.02, 0.55 - 0.5 * (clean / n) * min(1.0, n / 40))
        ref = f"mock-ft-{student_base}-{len(self._students)}"
        self._students[ref] = err
        return TrainedModel(ref=ref, base=student_base, cost_usd=round(0.002 * n, 4))

    def cost_per_call(self, model: str) -> float:
        return COSTS["teacher"] if self._error_rate(model) == 0.0 else COSTS["student"]
