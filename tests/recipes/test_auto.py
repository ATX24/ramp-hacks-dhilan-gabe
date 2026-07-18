"""Recipe-layer auto resolution tests."""

from __future__ import annotations

import pytest

from distillery.contracts.errors import DistilleryError, DistilleryErrorCode
from distillery.contracts.recipes import RecipeId
from distillery.recipes.auto import (
    build_auto_input_from_flags,
    require_trainable_resolution,
    resolve_recipe,
)


def test_explicit_sequence_never_reinterpreted() -> None:
    record = resolve_recipe(RecipeId.SEQUENCE_V1)
    assert record.resolved == "sequence.v1"
    assert "explicit_request" in record.reasons


def test_catalog_only_fails() -> None:
    with pytest.raises(DistilleryError) as exc:
        resolve_recipe(RecipeId.ON_POLICY_GKD)
    assert exc.value.code is DistilleryErrorCode.RECIPE_NOT_IMPLEMENTED


def test_auto_do_not_distill_refuses_training() -> None:
    inp = build_auto_input_from_flags(cheaper_baseline_satisfies_gate=True)
    record = resolve_recipe("auto", auto_input=inp)
    assert record.resolved == "do_not_distill"
    with pytest.raises(DistilleryError) as exc:
        require_trainable_resolution(record)
    assert exc.value.code is DistilleryErrorCode.CAPABILITY_UNAVAILABLE


def test_auto_usable_responses_selects_sequence() -> None:
    inp = build_auto_input_from_flags(usable_responses_exist=True)
    record = resolve_recipe("auto", auto_input=inp)
    assert require_trainable_resolution(record) == "sequence.v1"
