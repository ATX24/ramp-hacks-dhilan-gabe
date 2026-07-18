"""Built-in technique adapters must match recipe objective contracts."""

from __future__ import annotations

from distillery.recipes.logit_v1 import LogitV1Config, LogitV1Recipe
from distillery.recipes.sequence_v1 import SequenceV1Config, SequenceV1Recipe
from distillery.techniques import TechniqueRegistry, TechniqueRequest


def test_sequence_objective_parity(
    registry: TechniqueRegistry, sequence_context, sequence_config
) -> None:
    plan = registry.plan(
        TechniqueRequest(
            technique_id="sequence.v1",
            version="1.0.0",
            config=sequence_config,
        ),
        sequence_context,
    )
    recipe_config = SequenceV1Config(
        max_length=512,
        max_completion=160,
        require_nonempty_response=True,
        require_json_object_response=True,
        pad_token_id=None,
    )
    recipe_fields = SequenceV1Recipe(recipe_config).objective_fields()
    assert dict(plan.objective_fields) == recipe_fields
    assert dict(plan.adapter_config) == recipe_config.model_dump(mode="json")
    assert plan.loss.objective == recipe_fields["objective"]
    assert plan.loss.model_dump(mode="json")["fields"] == recipe_fields
    assert plan.training_load_plan is not None
    student = plan.training_load_plan.student.ref
    assert student.model_id == sequence_config["student_model_id"]
    assert student.revision == sequence_config["student_revision"]
    assert student.tokenizer_sha256 == sequence_config["student_tokenizer_sha256"]
    assert student.chat_template_sha256 == sequence_config["student_chat_template_sha256"]


def test_logit_objective_parity(registry: TechniqueRegistry, logit_context, logit_config) -> None:
    plan = registry.plan(
        TechniqueRequest(
            technique_id="logit.v1",
            version="1.0.0",
            config=logit_config,
        ),
        logit_context,
    )
    recipe_config = LogitV1Config(
        temperature=2.0,
        kd_weight=0.7,
        hard_ce_weight=0.3,
        vocab_chunk_size=4096,
        max_completion=160,
    )
    recipe_fields = LogitV1Recipe(recipe_config).objective_fields()
    assert dict(plan.objective_fields) == recipe_fields
    assert dict(plan.adapter_config) == {
        **recipe_config.model_dump(mode="json"),
        "max_length": 512,
    }
    assert plan.loss.temperature == recipe_fields["temperature"]
    assert plan.loss.kd_weight == recipe_fields["kd_weight"]
    assert plan.loss.hard_ce_weight == recipe_fields["hard_ce_weight"]
    assert plan.training_load_plan is not None
    assert plan.training_load_plan.model_dump(mode="json")["teacher"]["ref"] == {
        "model_id": logit_config["teacher_model_id"],
        "revision": logit_config["teacher_revision"],
        "tokenizer_sha256": logit_config["teacher_tokenizer_sha256"],
        "chat_template_sha256": logit_config["teacher_chat_template_sha256"],
    }
