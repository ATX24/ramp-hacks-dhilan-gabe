"""Recipe capability catalog and transparent auto resolver."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from pydantic import Field, StrictBool, StrictStr, model_validator

from distillery.contracts.base import FrozenModel
from distillery.contracts.errors import DistilleryError, DistilleryErrorCode, ErrorPayload

AUTO_BASELINE_PRECEDENCE_REASON = (
    "cheaper_baseline_satisfies_quality_and_economics_gate"
)
AUTO_SEQUENCE_RESPONSES_REASONS: tuple[str, ...] = (
    "usable_responses_present",
    "no_teacher_calls_required",
)
AUTO_LOGIT_REASONS: tuple[str, ...] = (
    "no_usable_responses",
    "local_white_box",
    "tokenizer_special_token_chat_template_compatible",
    "memory_dry_run_ok",
)
AUTO_SEQUENCE_TEACHER_REASONS: tuple[str, ...] = (
    "no_usable_responses",
    "logit_unavailable",
    "allowed_teacher_within_cost_ceiling",
)
AUTO_RESOLUTION_PRECEDENCE: tuple[str, ...] = (
    "do_not_distill_if_cheaper_baseline_satisfies_quality_and_economics_gate",
    "sequence.v1_if_usable_responses_exist",
    "logit.v1_if_all_white_box_compatibility_and_memory_gates_pass",
    "sequence.v1_if_allowed_teacher_can_fill_within_cost_ceiling",
    "typed_failure",
)


class RecipeId(StrEnum):
    AUTO = "auto"
    SEQUENCE_V1 = "sequence.v1"
    LOGIT_V1 = "logit.v1"
    # Catalog-only (not implemented)
    ON_POLICY_GKD = "on_policy_gkd"
    REVERSE_KL = "reverse_kl"
    JSD = "jsd"
    HIDDEN_STATE = "hidden_state"
    ATTENTION_RELATION = "attention_relation"
    PROGRESSIVE = "progressive"
    CURRICULUM = "curriculum"
    SELF_DISTILLATION = "self_distillation"
    MULTI_TEACHER = "multi_teacher"
    PREFERENCE_REWARD = "preference_reward"
    VERIFIED_REASONING = "verified_reasoning"
    CROSS_TOKENIZER_LOGIT = "cross_tokenizer_logit"
    DATA_FREE = "data_free"
    DATASET_CONDENSATION = "dataset_condensation"
    AGENT_TRAJECTORY = "agent_trajectory"


class RecipeStatus(StrEnum):
    IMPLEMENTED = "implemented"
    CATALOG_ONLY = "catalog_only"
    RESOLVER = "resolver"


class RecipeCapability(FrozenModel):
    recipe_id: RecipeId
    status: RecipeStatus
    signal: StrictStr
    access: StrictStr
    sampling: StrictStr
    teacher_topology: StrictStr
    data_regime: StrictStr
    schedule: StrictStr
    objective: StrictStr
    adaptation: StrictStr
    summary: StrictStr
    hard_gates: tuple[StrictStr, ...]


IMPLEMENTED_RECIPES: frozenset[RecipeId] = frozenset(
    {RecipeId.SEQUENCE_V1, RecipeId.LOGIT_V1}
)

CATALOG_ONLY_RECIPES: frozenset[RecipeId] = frozenset(
    {
        RecipeId.ON_POLICY_GKD,
        RecipeId.REVERSE_KL,
        RecipeId.JSD,
        RecipeId.HIDDEN_STATE,
        RecipeId.ATTENTION_RELATION,
        RecipeId.PROGRESSIVE,
        RecipeId.CURRICULUM,
        RecipeId.SELF_DISTILLATION,
        RecipeId.MULTI_TEACHER,
        RecipeId.PREFERENCE_REWARD,
        RecipeId.VERIFIED_REASONING,
        RecipeId.CROSS_TOKENIZER_LOGIT,
        RecipeId.DATA_FREE,
        RecipeId.DATASET_CONDENSATION,
        RecipeId.AGENT_TRAJECTORY,
    }
)

_recipe_catalog: dict[RecipeId, RecipeCapability] = {
    RecipeId.AUTO: RecipeCapability(
        recipe_id=RecipeId.AUTO,
        status=RecipeStatus.RESOLVER,
        signal="delegated",
        access="delegated",
        sampling="delegated",
        teacher_topology="delegated",
        data_regime="delegated",
        schedule="fixed",
        objective="delegated",
        adaptation="delegated",
        summary="Transparent resolver; never trains under an ambiguous method.",
        hard_gates=("never_ambiguous", "no_silent_downgrade"),
    ),
    RecipeId.SEQUENCE_V1: RecipeCapability(
        recipe_id=RecipeId.SEQUENCE_V1,
        status=RecipeStatus.IMPLEMENTED,
        signal="hard_target_sequence",
        access="black_box_response",
        sampling="offline_teacher_forced",
        teacher_topology="one_or_none",
        data_regime="observed_or_synthetic",
        schedule="fixed",
        objective="ce",
        adaptation="qlora",
        summary=(
            "Completion-only SFT/QLoRA on imported or teacher-generated responses; "
            "student retokenizes text."
        ),
        hard_gates=(
            "usable_responses_or_allowed_teacher",
            "schema_valid_text",
            "pinned_student_revision",
            "license_output_use",
            "memory_dry_run",
        ),
    ),
    RecipeId.LOGIT_V1: RecipeCapability(
        recipe_id=RecipeId.LOGIT_V1,
        status=RecipeStatus.IMPLEMENTED,
        signal="full_logits",
        access="local_white_box",
        sampling="offline_teacher_forced",
        teacher_topology="one",
        data_regime="observed_or_synthetic",
        schedule="fixed",
        objective="forward_kl_plus_hard_ce",
        adaptation="qlora",
        summary=(
            "Local white-box forward KL on teacher-forced completion positions "
            "with hard-target CE mixing."
        ),
        hard_gates=(
            "exact_tokenizer_fingerprint",
            "special_token_map_match",
            "compatible_chat_template",
            "local_teacher_weights",
            "full_distribution_implementation",
            "bounded_lengths",
            "memory_dry_run",
        ),
    ),
}

for _rid in CATALOG_ONLY_RECIPES:
    if _rid is RecipeId.AGENT_TRAJECTORY:
        summary = (
            "Catalog stub for Finance Agent role-masked trajectory supervision. "
            "agent_trajectory.v1 has an isolated objective/collator contract but no "
            "BYODT registration or training artifact; integration is pending review."
        )
    else:
        summary = f"Cataloged capability {_rid.value}; not implemented in MVP."
    _recipe_catalog[_rid] = RecipeCapability(
        recipe_id=_rid,
        status=RecipeStatus.CATALOG_ONLY,
        signal="catalog",
        access="catalog",
        sampling="catalog",
        teacher_topology="catalog",
        data_regime="catalog",
        schedule="catalog",
        objective="catalog",
        adaptation="catalog",
        summary=summary,
        hard_gates=("returns_RECIPE_NOT_IMPLEMENTED",),
    )
RECIPE_CATALOG: Mapping[RecipeId, RecipeCapability] = MappingProxyType(_recipe_catalog)
del _recipe_catalog


ResolvedRecipe = Literal["sequence.v1", "logit.v1", "do_not_distill"]
RequestedRecipe = Literal["auto", "sequence.v1", "logit.v1"]


class AutoResolverInput(FrozenModel):
    """Inputs for the locked auto resolver truth table."""

    cheaper_baseline_satisfies_gate: StrictBool = False
    usable_responses_exist: StrictBool = False
    local_white_box: StrictBool | None = None
    tokenizer_fingerprint_match: StrictBool | None = None
    special_token_map_match: StrictBool | None = None
    chat_template_compatible: StrictBool | None = None
    memory_dry_run_ok: StrictBool | None = None
    allowed_teacher_can_fill_within_ceiling: StrictBool = False


class AutoResolverResult(FrozenModel):
    resolved: ResolvedRecipe | None
    reasons: tuple[StrictStr, ...] = Field(min_length=1)
    rejected_alternatives: tuple[StrictStr, ...] = ()
    error_code: DistilleryErrorCode | None = None

    @model_validator(mode="after")
    def _resolution_matches_error(self) -> AutoResolverResult:
        if self.resolved is None and self.error_code is None:
            raise ValueError("an unresolved recipe requires a typed error_code")
        if self.resolved is not None and self.error_code is not None:
            raise ValueError("a resolved recipe cannot carry error_code")
        if (
            self.resolved == "do_not_distill"
            and AUTO_BASELINE_PRECEDENCE_REASON not in self.reasons
        ):
            raise ValueError(
                "do_not_distill requires the locked cheaper-baseline precedence reason"
            )
        return self


def resolve_auto_recipe(inp: AutoResolverInput) -> AutoResolverResult:
    """Resolve auto using the locked baseline-first precedence and no silent downgrade."""
    rejected: list[str] = []

    if inp.cheaper_baseline_satisfies_gate:
        return AutoResolverResult(
            resolved="do_not_distill",
            reasons=(AUTO_BASELINE_PRECEDENCE_REASON,),
            rejected_alternatives=("sequence.v1", "logit.v1"),
        )

    if inp.usable_responses_exist:
        return AutoResolverResult(
            resolved="sequence.v1",
            reasons=AUTO_SEQUENCE_RESPONSES_REASONS,
            rejected_alternatives=("logit.v1", "do_not_distill"),
        )

    logit_ok = (
        inp.local_white_box is True
        and inp.tokenizer_fingerprint_match is True
        and inp.special_token_map_match is True
        and inp.chat_template_compatible is True
        and inp.memory_dry_run_ok is True
    )
    if logit_ok:
        return AutoResolverResult(
            resolved="logit.v1",
            reasons=AUTO_LOGIT_REASONS,
            rejected_alternatives=("sequence.v1", "do_not_distill"),
        )
    rejected.append("logit.v1:capability_gates_failed")

    if inp.allowed_teacher_can_fill_within_ceiling:
        return AutoResolverResult(
            resolved="sequence.v1",
            reasons=AUTO_SEQUENCE_TEACHER_REASONS,
            rejected_alternatives=("logit.v1", "do_not_distill"),
        )
    rejected.append("sequence.v1:no_teacher_within_ceiling")

    return AutoResolverResult(
        resolved=None,
        reasons=("no_applicable_recipe",),
        rejected_alternatives=tuple(rejected),
        error_code=DistilleryErrorCode.AUTO_RESOLVER_FAILED,
    )


def validate_recipe_resolution(
    requested: RequestedRecipe,
    resolved: ResolvedRecipe,
    reasons: tuple[str, ...],
) -> None:
    """Validate the auditable no-silent-downgrade invariant."""
    if requested == "sequence.v1" and resolved != "sequence.v1":
        raise ValueError("explicit sequence.v1 must resolve only to sequence.v1")
    if requested == "logit.v1" and resolved != "logit.v1":
        raise ValueError("explicit logit.v1 must resolve only to logit.v1")
    if requested in {"sequence.v1", "logit.v1"}:
        if reasons != ("explicit_request",):
            raise ValueError("explicit recipes require resolver_reasons=('explicit_request',)")
        return
    if not reasons:
        raise ValueError("auto resolution requires at least one resolver reason")
    if "explicit_request" in reasons:
        raise ValueError("auto resolution cannot use the explicit_request reason")
    if (
        resolved == "do_not_distill"
        and reasons != (AUTO_BASELINE_PRECEDENCE_REASON,)
    ):
        raise ValueError(
            "auto do_not_distill requires the locked cheaper-baseline precedence reason"
        )
    allowed_reasons: dict[ResolvedRecipe, tuple[tuple[str, ...], ...]] = {
        "sequence.v1": (
            AUTO_SEQUENCE_RESPONSES_REASONS,
            AUTO_SEQUENCE_TEACHER_REASONS,
        ),
        "logit.v1": (AUTO_LOGIT_REASONS,),
        "do_not_distill": ((AUTO_BASELINE_PRECEDENCE_REASON,),),
    }
    if reasons not in allowed_reasons[resolved]:
        raise ValueError(
            f"auto {resolved} has non-canonical resolver_reasons {reasons!r}"
        )


def _validate_explicit_logit_capabilities(inp: AutoResolverInput) -> None:
    if (
        inp.tokenizer_fingerprint_match is False
        or inp.special_token_map_match is False
    ):
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.TOKENIZER_MISMATCH,
                "logit.v1 requires equal tokenizer fingerprints and special-token maps",
                details={
                    "tokenizer_fingerprint_match": inp.tokenizer_fingerprint_match,
                    "special_token_map_match": inp.special_token_map_match,
                },
            )
        )
    if inp.chat_template_compatible is False:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.CHAT_TEMPLATE_MISMATCH,
                "logit.v1 requires compatible chat-template semantics",
                details={
                    "chat_template_compatible": inp.chat_template_compatible,
                },
            )
        )
    if inp.memory_dry_run_ok is False:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.MEMORY_DRY_RUN_FAILED,
                "logit.v1 memory dry-run failed",
                details={"memory_dry_run_ok": inp.memory_dry_run_ok},
            )
        )
    if inp.local_white_box is not True:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.CAPABILITY_UNAVAILABLE,
                "logit.v1 requires known local white-box teacher and student access",
                details={"capability": "local_white_box"},
            )
        )
    if (
        inp.tokenizer_fingerprint_match is not True
        or inp.special_token_map_match is not True
    ):
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.TOKENIZER_MISMATCH,
                "logit.v1 tokenizer compatibility evidence is incomplete",
                details={
                    "tokenizer_fingerprint_match": inp.tokenizer_fingerprint_match,
                    "special_token_map_match": inp.special_token_map_match,
                },
            )
        )
    if inp.chat_template_compatible is not True:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.CHAT_TEMPLATE_MISMATCH,
                "logit.v1 chat-template compatibility evidence is incomplete",
                details={
                    "chat_template_compatible": inp.chat_template_compatible,
                },
            )
        )
    if inp.memory_dry_run_ok is not True:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.MEMORY_DRY_RUN_FAILED,
                "logit.v1 memory dry-run evidence is incomplete",
                details={"memory_dry_run_ok": inp.memory_dry_run_ok},
            )
        )


def resolve_requested_recipe(
    requested: RecipeId | str,
    *,
    auto_input: AutoResolverInput | None = None,
) -> AutoResolverResult:
    """
    Resolve a requested recipe.

    Explicit sequence.v1 / logit.v1 are never reinterpreted.
    Catalog-only recipes fail with RECIPE_NOT_IMPLEMENTED.
    """
    try:
        recipe = RecipeId(requested) if not isinstance(requested, RecipeId) else requested
    except (TypeError, ValueError) as exc:
        requested_value = requested if isinstance(requested, str) else type(requested).__name__
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_NOT_IMPLEMENTED,
                f"unknown recipe {requested_value!r}",
                details={"recipe": requested_value},
            )
        ) from exc

    if recipe in CATALOG_ONLY_RECIPES:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_NOT_IMPLEMENTED,
                f"recipe {recipe.value} is catalog-only and not implemented",
                details={"recipe": recipe.value, "status": RecipeStatus.CATALOG_ONLY.value},
            )
        )

    if recipe is RecipeId.SEQUENCE_V1:
        return AutoResolverResult(
            resolved="sequence.v1",
            reasons=("explicit_request",),
            rejected_alternatives=("auto", "logit.v1", "do_not_distill"),
        )

    if recipe is RecipeId.LOGIT_V1:
        # Method resolution stays pure for the current training adapter. Any supplied
        # claims are validated here; sealed manifests always require full evidence.
        if auto_input is not None:
            _validate_explicit_logit_capabilities(auto_input)
        return AutoResolverResult(
            resolved="logit.v1",
            reasons=("explicit_request",),
            rejected_alternatives=("auto", "sequence.v1", "do_not_distill"),
        )

    if recipe is RecipeId.AUTO:
        if auto_input is None:
            raise DistilleryError(
                ErrorPayload.from_code(
                    DistilleryErrorCode.CAPABILITY_UNAVAILABLE,
                    "auto resolver requires AutoResolverInput",
                    details={"recipe": "auto"},
                )
            )
        return resolve_auto_recipe(auto_input)

    raise DistilleryError(
        ErrorPayload.from_code(
            DistilleryErrorCode.RECIPE_NOT_IMPLEMENTED,
            f"unknown recipe {recipe.value}",
            details={"recipe": recipe.value},
        )
    )


# Truth-table rows used by contract tests (locked).
AUTO_RESOLVER_TRUTH_TABLE: tuple[tuple[AutoResolverInput, ResolvedRecipe | None], ...] = (
    (
        AutoResolverInput(cheaper_baseline_satisfies_gate=True, usable_responses_exist=True),
        "do_not_distill",
    ),
    (
        AutoResolverInput(usable_responses_exist=True),
        "sequence.v1",
    ),
    (
        AutoResolverInput(
            local_white_box=True,
            tokenizer_fingerprint_match=True,
            special_token_map_match=True,
            chat_template_compatible=True,
            memory_dry_run_ok=True,
        ),
        "logit.v1",
    ),
    (
        AutoResolverInput(
            local_white_box=True,
            tokenizer_fingerprint_match=False,
            allowed_teacher_can_fill_within_ceiling=True,
        ),
        "sequence.v1",
    ),
    (
        AutoResolverInput(
            local_white_box=True,
            tokenizer_fingerprint_match=True,
            special_token_map_match=True,
            chat_template_compatible=True,
            memory_dry_run_ok=False,
            allowed_teacher_can_fill_within_ceiling=True,
        ),
        "sequence.v1",
    ),
    (
        AutoResolverInput(),
        None,
    ),
)

__all__ = [
    "AUTO_BASELINE_PRECEDENCE_REASON",
    "AUTO_LOGIT_REASONS",
    "AUTO_RESOLUTION_PRECEDENCE",
    "AUTO_RESOLVER_TRUTH_TABLE",
    "AUTO_SEQUENCE_RESPONSES_REASONS",
    "AUTO_SEQUENCE_TEACHER_REASONS",
    "CATALOG_ONLY_RECIPES",
    "IMPLEMENTED_RECIPES",
    "RECIPE_CATALOG",
    "AutoResolverInput",
    "AutoResolverResult",
    "RecipeCapability",
    "RecipeId",
    "RecipeStatus",
    "RequestedRecipe",
    "ResolvedRecipe",
    "resolve_auto_recipe",
    "resolve_requested_recipe",
    "validate_recipe_resolution",
]
