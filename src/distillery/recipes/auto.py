"""Transparent auto recipe resolution for the trainer path."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from distillery.contracts.errors import DistilleryError, DistilleryErrorCode, ErrorPayload
from distillery.contracts.recipes import (
    AutoResolverInput,
    AutoResolverResult,
    RecipeId,
    resolve_auto_recipe,
    resolve_requested_recipe,
)


class AutoResolutionRecord(BaseModel):
    """Complete, auditable resolution record (never trains under ambiguity)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requested: str
    resolved: str | None
    reasons: tuple[str, ...]
    rejected_alternatives: tuple[str, ...] = ()
    error_code: DistilleryErrorCode | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)


def resolve_recipe(
    requested: RecipeId | str,
    *,
    auto_input: AutoResolverInput | None = None,
) -> AutoResolutionRecord:
    """
    Resolve a requested recipe into an explicit versioned method or do_not_distill.

    Explicit sequence.v1 / logit.v1 are never reinterpreted.
    Catalog-only recipes fail with RECIPE_NOT_IMPLEMENTED.
    """
    result: AutoResolverResult = resolve_requested_recipe(
        requested,
        auto_input=auto_input,
    )
    requested_value = (
        requested.value if isinstance(requested, RecipeId) else str(requested)
    )
    inputs = auto_input.model_dump(mode="json") if auto_input is not None else {}
    return AutoResolutionRecord(
        requested=requested_value,
        resolved=result.resolved,
        reasons=result.reasons,
        rejected_alternatives=result.rejected_alternatives,
        error_code=result.error_code,
        inputs=inputs,
    )


def require_trainable_resolution(record: AutoResolutionRecord) -> str:
    """Return a concrete trainable recipe id or raise a typed error."""
    if record.error_code is not None or record.resolved is None:
        raise DistilleryError(
            ErrorPayload.from_code(
                record.error_code or DistilleryErrorCode.AUTO_RESOLVER_FAILED,
                "auto resolver produced no trainable recipe",
                details={
                    "requested": record.requested,
                    "reasons": list(record.reasons),
                    "rejected_alternatives": list(record.rejected_alternatives),
                },
            )
        )
    if record.resolved == "do_not_distill":
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.CAPABILITY_UNAVAILABLE,
                "resolver recommends do_not_distill; refusing training path",
                details={
                    "requested": record.requested,
                    "reasons": list(record.reasons),
                },
            )
        )
    return record.resolved


def build_auto_input_from_flags(
    *,
    cheaper_baseline_satisfies_gate: bool = False,
    usable_responses_exist: bool = False,
    local_white_box: bool = False,
    tokenizer_fingerprint_match: bool = False,
    special_token_map_match: bool = False,
    chat_template_compatible: bool = False,
    memory_dry_run_ok: bool = False,
    allowed_teacher_can_fill_within_ceiling: bool = False,
) -> AutoResolverInput:
    """Construct the locked AutoResolverInput for the truth table."""
    return AutoResolverInput(
        cheaper_baseline_satisfies_gate=cheaper_baseline_satisfies_gate,
        usable_responses_exist=usable_responses_exist,
        local_white_box=local_white_box,
        tokenizer_fingerprint_match=tokenizer_fingerprint_match,
        special_token_map_match=special_token_map_match,
        chat_template_compatible=chat_template_compatible,
        memory_dry_run_ok=memory_dry_run_ok,
        allowed_teacher_can_fill_within_ceiling=allowed_teacher_can_fill_within_ceiling,
    )


# Re-export for recipe-layer callers that should not import contracts directly.
__all__ = [
    "AutoResolutionRecord",
    "build_auto_input_from_flags",
    "require_trainable_resolution",
    "resolve_auto_recipe",
    "resolve_recipe",
]
