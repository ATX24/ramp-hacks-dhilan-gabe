"""Built-in technique adapters must match recipe objective contracts."""

from __future__ import annotations

from distillery.recipes.logit_v1 import LogitV1Config, LogitV1Recipe
from distillery.recipes.sequence_v1 import SequenceV1Config, SequenceV1Recipe
from distillery.techniques import TechniqueRegistry, TechniqueRequest


def test_sequence_objective_parity(registry: TechniqueRegistry, sequence_context) -> None:
    plan = registry.plan(
        TechniqueRequest(
            technique_id="sequence.v1",
            version="1.0.0",
            config={
                "max_length": 512,
                "max_completion": 160,
                "seed": 17,
            },
        ),
        sequence_context,
    )
    recipe_fields = SequenceV1Recipe(SequenceV1Config()).objective_fields()
    assert plan.objective_fields["recipe_id"] == recipe_fields["recipe_id"]
    assert plan.objective_fields["mode"] == recipe_fields["mode"]
    assert plan.objective_fields["objective"] == recipe_fields["objective"]
    assert plan.objective_fields["signal"] == recipe_fields["signal"]
    assert plan.loss.objective == recipe_fields["objective"]


def test_logit_objective_parity(registry: TechniqueRegistry, logit_context) -> None:
    plan = registry.plan(
        TechniqueRequest(
            technique_id="logit.v1",
            version="1.0.0",
            config={
                "max_length": 512,
                "max_completion": 160,
                "seed": 17,
                "temperature": 2.0,
                "kd_weight": 0.7,
                "hard_ce_weight": 0.3,
            },
        ),
        logit_context,
    )
    recipe_fields = LogitV1Recipe(LogitV1Config()).objective_fields()
    assert plan.objective_fields["recipe_id"] == recipe_fields["recipe_id"]
    assert plan.objective_fields["mode"] == recipe_fields["mode"]
    assert plan.objective_fields["objective"] == recipe_fields["objective"]
    assert plan.objective_fields["signal"] == recipe_fields["signal"]
    assert plan.loss.temperature == recipe_fields["temperature"]
    assert plan.loss.kd_weight == recipe_fields["kd_weight"]
    assert plan.loss.hard_ce_weight == recipe_fields["hard_ce_weight"]
