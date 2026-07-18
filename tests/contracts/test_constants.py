"""Locked constant definitions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distillery.contracts.budgets import (
    PRIMARY_INDEX_WEIGHTS,
    EvaluationBudget,
    ProofGates,
    SmokeTrainingBudget,
    TrainingBudget,
)
from distillery.contracts.errors import ERROR_CODES, DistilleryErrorCode
from distillery.contracts.proof import ProofStatus
from distillery.contracts.recipes import (
    AUTO_BASELINE_PRECEDENCE_REASON,
    AUTO_RESOLUTION_PRECEDENCE,
    CATALOG_ONLY_RECIPES,
    IMPLEMENTED_RECIPES,
    RECIPE_CATALOG,
    AutoResolverInput,
    RecipeId,
    RecipeStatus,
    resolve_auto_recipe,
)
from distillery.contracts.states import ALLOWED_TRANSITIONS, TERMINAL_STATES, RunState


def test_primary_index_weights_locked() -> None:
    assert PRIMARY_INDEX_WEIGHTS.transaction_joint_exact == 0.45
    assert PRIMARY_INDEX_WEIGHTS.variance_joint_exact == 0.45
    assert PRIMARY_INDEX_WEIGHTS.json_schema_validity == 0.10
    total = (
        PRIMARY_INDEX_WEIGHTS.transaction_joint_exact
        + PRIMARY_INDEX_WEIGHTS.variance_joint_exact
        + PRIMARY_INDEX_WEIGHTS.json_schema_validity
    )
    assert abs(total - 1.0) < 1e-12


def test_proof_statuses_fixed() -> None:
    assert {s.value for s in ProofStatus} == {
        "proved",
        "do_not_distill",
        "failed_quality",
        "failed_economics",
        "insufficient_evidence",
    }


def test_error_codes_minimum_set() -> None:
    required = {
        "INVALID_DATASET",
        "SCHEMA_MISMATCH",
        "DATA_LEAKAGE_DETECTED",
        "UNSUPPORTED_LABEL_SOURCE",
        "MODEL_REVISION_UNPINNED",
        "TOKENIZER_MISMATCH",
        "CHAT_TEMPLATE_MISMATCH",
        "LICENSE_GATE_UNRESOLVED",
        "OUTPUT_USE_NOT_ALLOWED",
        "RECIPE_NOT_IMPLEMENTED",
        "RECIPE_INCOMPATIBLE",
        "CAPABILITY_UNAVAILABLE",
        "MEMORY_DRY_RUN_FAILED",
        "ESTIMATED_BUDGET_EXCEEDED",
        "AWS_QUOTA_UNAVAILABLE",
        "AWS_SUBMISSION_FAILED",
        "AWS_JOB_FAILED",
        "RUN_TIMEOUT",
        "CANCELLED",
        "ARTIFACT_INTEGRITY_FAILED",
        "EVALUATION_INCOMPLETE",
        "INSUFFICIENT_EVIDENCE",
    }
    assert required <= ERROR_CODES
    assert DistilleryErrorCode.RECIPE_NOT_IMPLEMENTED.value in ERROR_CODES


def test_recipe_catalog_implemented_and_catalog_only() -> None:
    assert IMPLEMENTED_RECIPES == {RecipeId.SEQUENCE_V1, RecipeId.LOGIT_V1}
    assert RecipeId.AUTO in RECIPE_CATALOG
    assert RECIPE_CATALOG[RecipeId.AUTO].status is RecipeStatus.RESOLVER
    for rid in IMPLEMENTED_RECIPES:
        assert RECIPE_CATALOG[rid].status is RecipeStatus.IMPLEMENTED
    for rid in CATALOG_ONLY_RECIPES:
        assert RECIPE_CATALOG[rid].status is RecipeStatus.CATALOG_ONLY
    assert RecipeId.CROSS_TOKENIZER_LOGIT in CATALOG_ONLY_RECIPES


def test_run_state_machine_shape() -> None:
    assert RunState.QUEUED in ALLOWED_TRANSITIONS
    assert TERMINAL_STATES == {
        RunState.SUCCEEDED,
        RunState.FAILED,
        RunState.CANCELLED,
    }
    for terminal in TERMINAL_STATES:
        assert ALLOWED_TRANSITIONS[terminal] == frozenset()
    # Happy path includes optional SYNTHESIZING skip from STARTING -> TRAINING
    assert RunState.SYNTHESIZING in ALLOWED_TRANSITIONS[RunState.STARTING]
    assert RunState.TRAINING in ALLOWED_TRANSITIONS[RunState.STARTING]
    assert RunState.CANCELLED not in ALLOWED_TRANSITIONS[RunState.FINALIZING]


def test_auto_baseline_precedence_is_locked() -> None:
    # The locked baseline gate runs before any billable training decision.
    assert AUTO_RESOLUTION_PRECEDENCE[0].startswith("do_not_distill")
    assert (
        AUTO_BASELINE_PRECEDENCE_REASON
        == "cheaper_baseline_satisfies_quality_and_economics_gate"
    )
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


def test_budget_defaults() -> None:
    smoke = SmokeTrainingBudget()
    full = TrainingBudget()
    eval_b = EvaluationBudget()
    gates = ProofGates()
    assert smoke.max_steps == 30
    assert full.max_steps == 200
    assert full.default_max_run_usd == 25.0
    assert eval_b.mixture_transaction_review == 0.45
    assert eval_b.mixture_variance_analysis == 0.45
    assert eval_b.mixture_cash_reconciliation == 0.10
    assert gates.required_seed_screen == 17
    assert gates.required_seed_replication == 23


def test_exported_catalog_and_transition_tables_are_immutable() -> None:
    with pytest.raises(TypeError):
        RECIPE_CATALOG[RecipeId.AUTO] = RECIPE_CATALOG[  # type: ignore[index]
            RecipeId.SEQUENCE_V1
        ]
    with pytest.raises(TypeError):
        ALLOWED_TRANSITIONS[RunState.QUEUED] = frozenset()  # type: ignore[index]
    with pytest.raises(AttributeError):
        ALLOWED_TRANSITIONS[RunState.QUEUED].add(RunState.TRAINING)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("budget_type", "updates"),
    [
        (SmokeTrainingBudget, {"max_steps": 0}),
        (SmokeTrainingBudget, {"max_completion": -1}),
        (SmokeTrainingBudget, {"kd_weight": 0.6, "hard_ce_weight": 0.3}),
        (TrainingBudget, {"train_examples": -1}),
        (TrainingBudget, {"vocab_chunk": 0}),
        (TrainingBudget, {"kd_weight": 0.4, "hard_ce_weight": 0.4}),
        (EvaluationBudget, {"batch_sizes": (1, 0)}),
    ],
)
def test_budgets_reject_invalid_counts_and_objective_weights(
    budget_type: type,
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        budget_type(**updates)
