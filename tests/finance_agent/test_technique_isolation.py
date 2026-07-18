"""agent_trajectory.v1 stays isolated from BYODT builtins and sequence/logit aliases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillery.finance_agent.technique import (
    AGENT_TRAJECTORY_TECHNIQUE_ID,
    AgentTrajectoryPlanAdapter,
    assert_not_sequence_or_logit_alias,
)
from distillery.finance_agent.technique.adapter import is_registered_in_byodt_builtins

ROOT = Path(__file__).resolve().parents[2]


def test_not_registered_in_builtins() -> None:
    assert is_registered_in_byodt_builtins() is False
    builtins_src = (ROOT / "src/distillery/techniques/builtins.py").read_text(encoding="utf-8")
    assert "agent_trajectory.v1" not in builtins_src
    assert "sequence.v1" in builtins_src
    assert "logit.v1" in builtins_src


def test_adapter_plans_with_distinct_technique_id() -> None:
    config = json.loads(
        (ROOT / "examples/byodt/agent_trajectory_v1/sample_config.json").read_text(
            encoding="utf-8"
        )
    )
    plan = AgentTrajectoryPlanAdapter().plan(config)
    assert plan.technique_id == AGENT_TRAJECTORY_TECHNIQUE_ID
    assert plan.byodt_integration == "pending_review"
    assert plan.objective == "trajectory_ce"
    assert plan.technique_id != "sequence.v1"
    assert plan.objective != "ce"


def test_alias_guard() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_not_sequence_or_logit_alias("sequence.v1")
    with pytest.raises(ValueError, match="forbidden"):
        assert_not_sequence_or_logit_alias("logit.v1")


def test_demo_contracts_exist() -> None:
    chat = json.loads(
        (ROOT / "examples/finance_agent/chat_demo_contract.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (ROOT / "examples/finance_agent/model_registry_finance_agent.json").read_text(
            encoding="utf-8"
        )
    )
    assert chat["mode_id"] == "finance_agent"
    assert chat["technique_id"] == "agent_trajectory.v1"
    assert chat["integration_status"] == "contract_only"
    assert registry["schema_version"] == "distillery.demo_model_registry.v1"
    assert registry["mode_id"] == "finance_agent"
    assert all(model["stats"]["recipe"] == "agent_trajectory.v1" for model in registry["models"])
