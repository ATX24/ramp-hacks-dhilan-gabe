"""Recipe catalog and capability matrix. Only sequence.v1 and logit.v1 are implemented."""
from __future__ import annotations

from pydantic import BaseModel

IMPLEMENTED = {"sequence.v1", "logit.v1"}

CATALOG_ONLY = {
    "gkd.on_policy.v0": "on-policy GKD",
    "reverse_kl.v0": "reverse KL / JSD objectives",
    "hidden_state.v0": "hidden-state distillation",
    "attention_relation.v0": "attention/relation distillation",
    "progressive.v0": "progressive/curriculum distillation",
    "self_distill.v0": "self-distillation",
    "multi_teacher.v0": "multi-teacher/ensemble distillation",
    "preference.v0": "preference/reward distillation",
    "verified_reasoning.v0": "verified-reasoning distillation",
    "cross_tokenizer_logit.v0": "cross-tokenizer logit alignment",
    "data_free.v0": "data-free distillation",
    "condensation.v0": "dataset condensation",
    "agent_trajectory.v0": "agent-trajectory distillation",
}


class RecipeInfo(BaseModel):
    name: str
    implemented: bool
    signal: str
    access: str
    notes: str


def catalog() -> list[RecipeInfo]:
    items = [
        RecipeInfo(
            name="sequence.v1", implemented=True, signal="hard target sequence",
            access="black-box response",
            notes="Completion-only SFT/QLoRA on imported or teacher-generated responses. "
                  "Crosses tokenizer boundaries because the student retokenizes text.",
        ),
        RecipeInfo(
            name="logit.v1", implemented=True, signal="full logits (forward KL + hard CE)",
            access="local white-box",
            notes="Requires exactly matching tokenizer fingerprints and a local white-box teacher. "
                  "Full-vocabulary chunked forward KL; no top-K approximation.",
        ),
        RecipeInfo(
            name="auto", implemented=True, signal="resolver",
            access="n/a",
            notes="Transparent resolver delegating to exactly one versioned recipe, "
                  "or recommending do_not_distill. Never silently downgrades.",
        ),
    ]
    for name, desc in CATALOG_ONLY.items():
        items.append(RecipeInfo(name=name, implemented=False, signal=desc, access="n/a",
                                notes="Cataloged; returns RECIPE_NOT_IMPLEMENTED."))
    from ..recipes import custom as custom_mod  # late import: avoids cycle
    for name in custom_mod.custom_names():
        r = custom_mod.get_custom(name)
        items.append(RecipeInfo(
            name=name, implemented=True, signal="user-defined synthesis strategy",
            access="black-box response",
            notes=f"{r.description} Requires: {sorted(r.requires) or 'nothing'}. "
                  "Curation runs in the control plane; training executes sequence.v1 "
                  "on the recipe's output dataset."))
    return items


class RecipeResolution(BaseModel):
    requested: str
    resolved: str | None  # None => do_not_distill or failure
    resolver_reasons: list[str]
    rejected_alternatives: list[dict]
    do_not_distill: bool = False
