"""Registry resolve/plan through the public technique seam."""

from __future__ import annotations

import pytest

from distillery.techniques import (
    TechniqueError,
    TechniqueErrorCode,
    TechniqueRegistry,
    TechniqueRequest,
)


def test_builtins_registered(registry: TechniqueRegistry) -> None:
    keys = {d.technique_key for d in registry.list_techniques()}
    assert keys == {"logit.v1@1.0.0", "sequence.v1@1.0.0"}


def test_plan_sequence_through_registry(registry, sequence_context, sequence_config) -> None:
    plan = registry.plan(
        TechniqueRequest(
            technique_id="sequence.v1",
            version="1.0.0",
            config=sequence_config,
        ),
        sequence_context,
    )
    assert plan.lifecycle.value == "planned"
    assert plan.training_load_plan is not None
    assert plan.training_load_plan.recipe == "sequence.v1"
    assert plan.loss.objective == "ce"
    assert plan.external_execution is None
    assert plan.protocol_sha256


def test_plan_logit_through_registry(registry, logit_context, logit_config) -> None:
    plan = registry.plan(
        TechniqueRequest(
            technique_id="logit.v1",
            version="1.0.0",
            config=logit_config,
        ),
        logit_context,
    )
    assert plan.training_load_plan is not None
    assert plan.training_load_plan.teacher is not None
    assert plan.loss.signal == "full_logits"
    assert plan.objective_fields["objective"] == "forward_kl_plus_hard_ce"


def test_unknown_technique_no_silent_fallback(registry, sequence_context) -> None:
    with pytest.raises(TechniqueError) as excinfo:
        registry.plan(
            TechniqueRequest(
                technique_id="hackathon.missing.technique",
                version="1.0.0",
                config={"max_length": 512, "max_completion": 160, "seed": 1},
            ),
            sequence_context,
        )
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_UNKNOWN


def test_external_plan_yields_channel_not_import(
    registry, logit_context, external_descriptor, external_config
) -> None:
    registry.register(external_descriptor)
    plan = registry.plan(
        TechniqueRequest(
            technique_id="hackathon.dhilan.reverse_kl",
            version="1.0.0",
            config=external_config,
        ),
        logit_context,
    )
    assert plan.training_load_plan is None
    assert plan.external_execution is not None
    assert plan.external_execution.network_isolation_required is True
    assert plan.external_execution.channel_plan_filename == "technique_plan.json"
    assert plan.environment == logit_context
