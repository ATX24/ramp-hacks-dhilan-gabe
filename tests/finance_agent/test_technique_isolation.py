"""agent_trajectory.v1 objective identity, claims, and BYODT isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillery.finance_agent.generate import GeneratedAgentCorpus
from distillery.finance_agent.technique import (
    AGENT_TRAJECTORY_TECHNIQUE_ID,
    TRAJECTORY_RENDER_TEMPLATE_SHA256,
    AgentTrajectoryPlanAdapter,
    assert_not_sequence_or_logit_alias,
)
from distillery.finance_agent.technique.adapter import is_registered_in_byodt_builtins

ROOT = Path(__file__).resolve().parents[2]


def _config(corpus: GeneratedAgentCorpus) -> dict:
    manifest = corpus.manifest
    return {
        "max_length": 16_384,
        "max_supervised_tokens": 8_192,
        "pad_token_id": 0,
        "seed": 17,
        "corpus_seed": manifest["seed"],
        "trajectory_corpus_sha256": manifest["corpus_sha256"],
        "corpus_order_sha256": manifest["corpus_order_sha256"],
        "system_prompt_set_sha256": manifest["system_prompt_set_sha256"],
        "tool_schema_set_sha256": manifest["tool_schema_set_sha256"],
        "trajectory_render_template_sha256": TRAJECTORY_RENDER_TEMPLATE_SHA256,
        "mask_tool_results": True,
        "label_source": "oracle",
        "student_model_id": None,
        "student_revision": None,
        "tokenizer_sha256": None,
        "chat_template_sha256": None,
        "license_status": "unknown",
        "cost_status": "unknown",
    }


def test_not_registered_in_builtins() -> None:
    assert is_registered_in_byodt_builtins() is False
    builtins = (ROOT / "src/distillery/techniques/builtins.py").read_text(encoding="utf-8")
    assert "agent_trajectory.v1" not in builtins


def test_adapter_has_distinct_role_masked_objective_and_no_training_claim(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    plan = AgentTrajectoryPlanAdapter().plan(_config(smoke_corpus))
    assert plan.technique_id == AGENT_TRAJECTORY_TECHNIQUE_ID
    assert plan.technique_id != "sequence.v1"
    assert plan.objective == "role_masked_trajectory_ce"
    assert plan.supervision_source == "oracle_trajectory_labels"
    assert plan.collator_config.mask_tool_results is True
    assert plan.byodt_integration == "pending_review"
    assert plan.training_ready is False
    assert plan.training_status == "not_materialized"
    assert "training_path_not_wired" in plan.readiness_blockers


@pytest.mark.parametrize(
    "forbidden",
    [
        {"teacher_model_id": "unproven/model"},
        {"teacher_revision": "a" * 40},
        {"specialist_task": "policy"},
    ],
)
def test_adapter_rejects_unproven_teacher_and_specialist_claims(
    forbidden: dict,
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    with pytest.raises(ValueError):
        AgentTrajectoryPlanAdapter().plan({**_config(smoke_corpus), **forbidden})


def test_alias_guard() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_not_sequence_or_logit_alias("sequence.v1")
    with pytest.raises(ValueError, match="forbidden"):
        assert_not_sequence_or_logit_alias("logit.v1")


def test_sample_config_uses_real_corpus_bindings_and_no_placeholders(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    config = json.loads(
        (ROOT / "examples/byodt/agent_trajectory_v1/sample_config.json").read_text(encoding="utf-8")
    )
    manifest = smoke_corpus.manifest
    assert config["trajectory_corpus_sha256"] == manifest["corpus_sha256"]
    assert config["corpus_order_sha256"] == manifest["corpus_order_sha256"]
    assert config["system_prompt_set_sha256"] == manifest["system_prompt_set_sha256"]
    assert config["tool_schema_set_sha256"] == manifest["tool_schema_set_sha256"]
    AgentTrajectoryPlanAdapter().plan(config)


def test_demo_metadata_advertises_no_untrained_or_specialist_model() -> None:
    registry = json.loads(
        (ROOT / "examples/finance_agent/model_registry_finance_agent.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry["mode_id"] == "finance_agent"
    assert registry["availability"] == "not_materialized"
    assert registry["models"] == []
    paths = list((ROOT / "src/distillery/finance_agent").rglob("*.py"))
    paths.extend(
        (
            ROOT / "docs/finance-agent.md",
            ROOT / "docs/adr/0002-agent-trajectory-v1-technique.md",
            ROOT / "examples/byodt/agent_trajectory_v1/README.md",
            ROOT / "examples/byodt/agent_trajectory_v1/sample_config.json",
            ROOT / "examples/finance_agent/model_registry_finance_agent.json",
        )
    )
    scoped = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "72B" not in scoped
    assert "Qwen2.5-72B" not in scoped
