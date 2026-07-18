"""Recipe catalog and auto-resolver truth table."""

from __future__ import annotations

import pytest

from distillery.contracts.errors import DistilleryError, DistilleryErrorCode
from distillery.contracts.recipes import (
    AUTO_BASELINE_PRECEDENCE_REASON,
    AUTO_RESOLUTION_PRECEDENCE,
    AUTO_RESOLVER_TRUTH_TABLE,
    CATALOG_ONLY_RECIPES,
    AutoResolverInput,
    RecipeId,
    resolve_auto_recipe,
    resolve_requested_recipe,
)


@pytest.mark.parametrize(("inp", "expected"), AUTO_RESOLVER_TRUTH_TABLE)
def test_auto_resolver_truth_table(inp: AutoResolverInput, expected: str | None) -> None:
    result = resolve_auto_recipe(inp)
    assert result.resolved == expected
    if expected is None:
        assert result.error_code is DistilleryErrorCode.AUTO_RESOLVER_FAILED
        assert "no_applicable_recipe" in result.reasons
    else:
        assert result.error_code is None
        assert result.reasons


def test_do_not_distill_beats_usable_responses() -> None:
    result = resolve_auto_recipe(
        AutoResolverInput(
            cheaper_baseline_satisfies_gate=True,
            usable_responses_exist=True,
            local_white_box=True,
            tokenizer_fingerprint_match=True,
            special_token_map_match=True,
            chat_template_compatible=True,
            memory_dry_run_ok=True,
        )
    )
    assert result.resolved == "do_not_distill"
    assert result.reasons == (AUTO_BASELINE_PRECEDENCE_REASON,)
    assert AUTO_RESOLUTION_PRECEDENCE[0].startswith("do_not_distill")


def test_explicit_sequence_never_reinterpreted() -> None:
    # Even when logit would be preferred, explicit sequence stays sequence.
    result = resolve_requested_recipe(
        RecipeId.SEQUENCE_V1,
        auto_input=AutoResolverInput(
            local_white_box=True,
            tokenizer_fingerprint_match=True,
            special_token_map_match=True,
            chat_template_compatible=True,
            memory_dry_run_ok=True,
        ),
    )
    assert result.resolved == "sequence.v1"
    assert result.reasons == ("explicit_request",)


def test_explicit_logit_never_reinterpreted() -> None:
    result = resolve_requested_recipe(
        RecipeId.LOGIT_V1,
        auto_input=AutoResolverInput(
            usable_responses_exist=True,
            local_white_box=True,
            tokenizer_fingerprint_match=True,
            special_token_map_match=True,
            chat_template_compatible=True,
            memory_dry_run_ok=True,
        ),
    )
    assert result.resolved == "logit.v1"
    assert result.reasons == ("explicit_request",)


def test_auto_requires_input() -> None:
    with pytest.raises(DistilleryError) as exc:
        resolve_requested_recipe(RecipeId.AUTO)
    assert exc.value.code is DistilleryErrorCode.CAPABILITY_UNAVAILABLE


@pytest.mark.parametrize("recipe", sorted(CATALOG_ONLY_RECIPES, key=lambda r: r.value))
def test_catalog_only_raises_not_implemented(recipe: RecipeId) -> None:
    with pytest.raises(DistilleryError) as exc:
        resolve_requested_recipe(recipe)
    assert exc.value.code is DistilleryErrorCode.RECIPE_NOT_IMPLEMENTED
    assert not exc.value.payload.retryable
    assert exc.value.payload.details["recipe"] == recipe.value


def test_requested_string_forms_accepted() -> None:
    assert resolve_requested_recipe("sequence.v1").resolved == "sequence.v1"
    # This chooses a method only. SealedRunManifest requires full logit evidence.
    assert resolve_requested_recipe("logit.v1").resolved == "logit.v1"


def test_unknown_recipe_string_raises_typed_not_implemented() -> None:
    with pytest.raises(DistilleryError) as exc:
        resolve_requested_recipe("future.recipe.v9")
    assert exc.value.code is DistilleryErrorCode.RECIPE_NOT_IMPLEMENTED
    assert exc.value.payload.details == {"recipe": "future.recipe.v9"}


@pytest.mark.parametrize(
    ("capabilities", "expected_code"),
    [
        (
            AutoResolverInput(
                tokenizer_fingerprint_match=False,
                special_token_map_match=True,
            ),
            DistilleryErrorCode.TOKENIZER_MISMATCH,
        ),
        (
            AutoResolverInput(chat_template_compatible=False),
            DistilleryErrorCode.CHAT_TEMPLATE_MISMATCH,
        ),
        (
            AutoResolverInput(memory_dry_run_ok=False),
            DistilleryErrorCode.MEMORY_DRY_RUN_FAILED,
        ),
        (
            AutoResolverInput(local_white_box=False),
            DistilleryErrorCode.CAPABILITY_UNAVAILABLE,
        ),
    ],
)
def test_explicit_logit_preserves_typed_capability_failures(
    capabilities: AutoResolverInput,
    expected_code: DistilleryErrorCode,
) -> None:
    with pytest.raises(DistilleryError) as exc:
        resolve_requested_recipe("logit.v1", auto_input=capabilities)
    assert exc.value.code is expected_code


def test_explicit_logit_rejects_unknown_capability_claims_when_supplied() -> None:
    with pytest.raises(DistilleryError) as exc:
        resolve_requested_recipe(
            "logit.v1",
            auto_input=AutoResolverInput(local_white_box=True),
        )
    assert exc.value.code is DistilleryErrorCode.TOKENIZER_MISMATCH
