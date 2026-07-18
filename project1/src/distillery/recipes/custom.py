"""User-defined distillation recipes (Tinker-style).

A custom recipe is ordinary Python: a class with a name, declared capability
requirements, and run(ctx, examples) written against SynthesisContext
primitives. Recipes operate at the synthesis/curation stage — they decide
which examples get labeled, by whom, and which survive — and always produce a
NEW immutable dataset. Training remains sequence.v1 on the recipe's output,
and the proof gate is untouched: recipes never see test splits.

Register in-process via register(), or point DISTILLERY_RECIPES at a
comma-separated list of importable modules that call register() at import time.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Callable, Protocol

from ..contracts.dataset import Example, canonical_json
from ..contracts.errors import OutputUseNotAllowed, RecipeIncompatible
from ..data.validate import validate_output
from ..synthesis import teacher as teacher_mod


class SynthesisContext:
    """Primitives a recipe composes. Teacher spend is metered and capped;
    test splits are structurally out of reach (the caller never passes them)."""

    def __init__(self, out_dir: Path, max_cost_usd: float, dry_run: bool,
                 emit: Callable[[dict], None]):
        self.out_dir = out_dir
        self.max_cost_usd = max_cost_usd
        self.dry_run = dry_run
        self._emit = emit
        self.teacher_stats: list[dict] = []

    # -- labeling ----------------------------------------------------------
    def teacher_label(self, examples: list[Example]) -> dict:
        """Fill missing responses with the teacher (Claude Opus). Validated,
        provenance-stamped, appended to synthesis_responses.jsonl."""
        for ex in examples:
            if ex.provenance.split not in ("train", "validation"):
                raise OutputUseNotAllowed("Recipes may only label train/validation examples.",
                                          details={"example_id": ex.example_id})
        stats = teacher_mod.synthesize_missing(
            examples, self.out_dir / "synthesis_responses.jsonl",
            max_cost_usd=self.max_cost_usd, dry_run=self.dry_run)
        self.teacher_stats.append(stats)
        return stats

    def oracle_label(self, examples: list[Example]) -> int:
        return teacher_mod.materialize_oracle_responses(examples)

    # -- verification ------------------------------------------------------
    def is_valid(self, ex: Example) -> bool:
        """Deterministic schema/consistency validation of ex.response."""
        if not ex.response:
            return False
        _, errs = validate_output(ex.task, ex.response, ex.input)
        return not errs

    def agrees_with_oracle(self, ex: Example) -> bool:
        """Exact-match check of ex.response against the latent oracle's
        expected output. The strongest (and most expensive-to-earn) filter."""
        if not ex.response:
            return False
        return ex.response.strip() == canonical_json(ex.expected_output)

    def emit(self, event: str, **data) -> None:
        self._emit({"event": event, **data})


class CustomRecipe(Protocol):
    name: str
    requires: frozenset[str]  # subset of {"teacher", "oracle"}
    description: str

    def run(self, ctx: SynthesisContext, examples: list[Example]) -> list[Example]:
        """Return the curated, labeled examples that training should use."""
        ...


_REGISTRY: dict[str, CustomRecipe] = {}


def register(recipe: CustomRecipe) -> CustomRecipe:
    if recipe.name in {"auto", "sequence.v1", "logit.v1"}:
        raise ValueError(f"'{recipe.name}' shadows a built-in recipe")
    _REGISTRY[recipe.name] = recipe
    return recipe


def get_custom(name: str) -> CustomRecipe | None:
    return _REGISTRY.get(name)


def custom_names() -> list[str]:
    return sorted(_REGISTRY)


def check_requirements(recipe: CustomRecipe, *, teacher_available: bool) -> None:
    if "teacher" in recipe.requires and not teacher_available:
        raise RecipeIncompatible(
            f"Recipe '{recipe.name}' requires the teacher but ANTHROPIC_API_KEY is not set.",
            details={"recipe": recipe.name, "requires": sorted(recipe.requires)})


def load_plugins() -> list[str]:
    """Import modules named in DISTILLERY_RECIPES so their register() calls run."""
    loaded = []
    for mod in filter(None, os.environ.get("DISTILLERY_RECIPES", "").split(",")):
        importlib.import_module(mod.strip())
        loaded.append(mod.strip())
    return loaded


# ---------------------------------------------------------------------------
# Built-in custom recipes: written against the same primitives users get.


class RejectionSampling:
    name = "rejection_sampling.v1"
    requires = frozenset({"teacher"})
    description = ("Teacher-label everything, then keep only responses that pass "
                   "deterministic validation AND match the latent oracle. Smaller, "
                   "cleaner training set.")

    def run(self, ctx: SynthesisContext, examples: list[Example]) -> list[Example]:
        ctx.teacher_label(examples)
        kept = [ex for ex in examples if ctx.is_valid(ex) and ctx.agrees_with_oracle(ex)]
        ctx.emit("rejection_filter", kept=len(kept), dropped=len(examples) - len(kept))
        return kept


class OracleCurriculum:
    name = "oracle_curriculum.v1"
    requires = frozenset({"oracle"})
    description = ("Zero-teacher-cost baseline recipe: oracle-label everything, but "
                   "oversample hard examples 2x so the student sees a difficulty "
                   "curriculum. Proves recipes can beat uniform sampling for free.")

    def run(self, ctx: SynthesisContext, examples: list[Example]) -> list[Example]:
        ctx.oracle_label(examples)
        hard = [ex for ex in examples if ex.difficulty == "hard"]
        ctx.emit("curriculum", base=len(examples), hard_oversampled=len(hard))
        return examples + [ex.model_copy(deep=True) for ex in hard]


register(RejectionSampling())
register(OracleCurriculum())
