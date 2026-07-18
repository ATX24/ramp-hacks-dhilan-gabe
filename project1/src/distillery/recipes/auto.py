"""Transparent `auto` resolver. Delegates to exactly one versioned recipe or
recommends do_not_distill; never silently downgrades a requested recipe."""
from __future__ import annotations

from ..contracts.errors import RecipeNotImplemented, RecipeIncompatible
from ..contracts.recipes import IMPLEMENTED, CATALOG_ONLY, RecipeResolution


def resolve(requested: str, *, has_valid_responses: bool, teacher_access: str,
            tokenizers_match: bool, memory_dry_run_ok: bool,
            teacher_available: bool, baseline_meets_gate: bool = False) -> RecipeResolution:
    if requested in CATALOG_ONLY:
        raise RecipeNotImplemented(f"Recipe '{requested}' is cataloged but not implemented.",
                                   details={"recipe": requested})

    from .custom import check_requirements, get_custom
    custom = get_custom(requested)
    if custom is not None:
        check_requirements(custom, teacher_available=teacher_available)
        return RecipeResolution(
            requested=requested, resolved=requested,
            resolver_reasons=[f"user-defined recipe '{requested}' explicitly requested; "
                              "requirements satisfied; training arm is sequence.v1 on the "
                              "recipe's output dataset"],
            rejected_alternatives=[])

    if requested not in IMPLEMENTED and requested != "auto":
        raise RecipeNotImplemented(f"Unknown recipe '{requested}'.", details={"recipe": requested})

    reasons: list[str] = []
    rejected: list[dict] = []

    if requested == "logit.v1":
        if teacher_access != "white_box":
            raise RecipeIncompatible(
                "logit.v1 requires a local white-box teacher; the configured teacher is API black-box "
                "(Claude Opus). No silent downgrade — request sequence.v1 explicitly.",
                details={"teacher_access": teacher_access})
        if not tokenizers_match:
            raise RecipeIncompatible("TOKENIZER_MISMATCH for logit.v1.", details={})
        if not memory_dry_run_ok:
            raise RecipeIncompatible("MEMORY_DRY_RUN_FAILED for logit.v1.", details={})
        return RecipeResolution(requested=requested, resolved="logit.v1",
                                resolver_reasons=["explicitly requested and all gates pass"],
                                rejected_alternatives=[])

    if requested == "sequence.v1":
        if not has_valid_responses and not teacher_available:
            raise RecipeIncompatible("sequence.v1 needs usable responses or an available teacher.",
                                     details={})
        return RecipeResolution(requested=requested, resolved="sequence.v1",
                                resolver_reasons=["explicitly requested"], rejected_alternatives=[])

    # auto
    if baseline_meets_gate:
        reasons.append("a cheaper baseline already satisfies the proof gate")
        return RecipeResolution(requested="auto", resolved=None, resolver_reasons=reasons,
                                rejected_alternatives=rejected, do_not_distill=True)
    if has_valid_responses:
        reasons.append("valid responses already exist -> sequence.v1 with zero teacher calls")
        return RecipeResolution(requested="auto", resolved="sequence.v1",
                                resolver_reasons=reasons, rejected_alternatives=rejected)
    if teacher_access == "white_box" and tokenizers_match and memory_dry_run_ok:
        reasons.append("white-box teacher with matching tokenizer and passing memory dry-run")
        return RecipeResolution(requested="auto", resolved="logit.v1",
                                resolver_reasons=reasons, rejected_alternatives=rejected)
    rejected.append({"recipe": "logit.v1",
                     "reason": f"teacher_access={teacher_access}, tokenizers_match={tokenizers_match}"})
    if teacher_available:
        reasons.append("teacher (Claude Opus) can supply missing labels within the cost ceiling -> sequence.v1")
        return RecipeResolution(requested="auto", resolved="sequence.v1",
                                resolver_reasons=reasons, rejected_alternatives=rejected)
    raise RecipeIncompatible("auto could not resolve any recipe: no responses, no compatible "
                             "white-box pair, no available teacher.", details={"rejected": rejected})
