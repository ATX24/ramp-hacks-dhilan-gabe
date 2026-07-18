"""Plan-only adapter for agent_trajectory.v1.

This module is intentionally NOT imported by ``TechniqueRegistry.with_builtins()``.
BYODT integration happens only after review. Until then callers import this adapter
directly. It must never advertise itself as ``sequence.v1`` or ``logit.v1``.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictStr, model_validator

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import Sha256Hex, content_sha256
from distillery.finance_agent.contracts import TECHNIQUE_ID_AGENT_TRAJECTORY

AGENT_TRAJECTORY_TECHNIQUE_ID: Literal["agent_trajectory.v1"] = TECHNIQUE_ID_AGENT_TRAJECTORY
AGENT_TRAJECTORY_VERSION: Literal["1.0.0"] = "1.0.0"

_FORBIDDEN_ALIASES = frozenset({"sequence.v1", "logit.v1", "sequence", "logit"})

CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "max_length",
        "max_completion",
        "seed",
        "teacher_model_id",
        "teacher_revision",
        "student_model_id",
        "student_revision",
        "trajectory_corpus_sha256",
    ],
    "properties": {
        "max_length": {"type": "integer", "minimum": 2},
        "max_completion": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer", "minimum": 0},
        "teacher_model_id": {"type": "string", "minLength": 1},
        "teacher_revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "student_model_id": {"type": "string", "minLength": 1},
        "student_revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "trajectory_corpus_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "mask_tool_results": {"type": "boolean"},
        "specialist_task": {
            "type": "string",
            "enum": ["generalist", "policy", "ledger", "reconciliation", "variance"],
        },
    },
}


class AgentTrajectoryPlan(FrozenModel):
    """Sealed plan for sequence-level trajectory distillation.

    Distinct from ``distillery.technique.plan.v1`` until BYODT review wires it in.
    """

    schema_version: Literal["finance_agent.technique_plan.v1"] = (
        "finance_agent.technique_plan.v1"
    )
    technique_id: Literal["agent_trajectory.v1"] = AGENT_TRAJECTORY_TECHNIQUE_ID
    version: Literal["1.0.0"] = AGENT_TRAJECTORY_VERSION
    display_name: StrictStr = "Agent trajectory distillation v1"
    teacher_signal: Literal["trajectory_hard_targets"] = "trajectory_hard_targets"
    student_targets: Literal["tinyfable_generalist_and_specialists"] = (
        "tinyfable_generalist_and_specialists"
    )
    teacher_model_id: StrictStr
    teacher_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    student_model_id: StrictStr
    student_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: Sha256Hex
    protocol_sha256: Sha256Hex
    objective: Literal["trajectory_ce"] = "trajectory_ce"
    metrics: tuple[StrictStr, ...] = (
        "tool_selection_accuracy",
        "argument_exactness",
        "tool_result_use",
        "final_answer_correctness",
        "unnecessary_calls",
        "latency_ms",
        "cost_usd_micros",
        "end_to_end_success",
    )
    byodt_integration: Literal["pending_review"] = "pending_review"
    notes: StrictStr = (
        "Supervises full Finance Agent trajectories from a 72B teacher into "
        "TinyFable Generalist and task specialists. Not sequence.v1 / logit.v1."
    )
    objective_fields: FrozenJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_alias(self) -> AgentTrajectoryPlan:
        assert_not_sequence_or_logit_alias(self.technique_id)
        return self

    def plan_hash(self) -> str:
        return content_sha256(self.model_dump(mode="json"))


def assert_not_sequence_or_logit_alias(technique_id: str) -> None:
    if technique_id in _FORBIDDEN_ALIASES:
        raise ValueError(
            f"{technique_id!r} is forbidden; Finance Agent distillation must use "
            f"{AGENT_TRAJECTORY_TECHNIQUE_ID}"
        )
    if technique_id != AGENT_TRAJECTORY_TECHNIQUE_ID:
        raise ValueError(
            f"expected technique_id={AGENT_TRAJECTORY_TECHNIQUE_ID!r}, got {technique_id!r}"
        )


class AgentTrajectoryPlanAdapter:
    """Isolated plan adapter. Do not register in TechniqueRegistry until review."""

    technique_id = AGENT_TRAJECTORY_TECHNIQUE_ID
    version = AGENT_TRAJECTORY_VERSION

    def validate(self, config: Mapping[str, Any]) -> str:
        from jsonschema import Draft202012Validator

        assert_not_sequence_or_logit_alias(self.technique_id)
        errors = sorted(
            Draft202012Validator(CONFIG_SCHEMA).iter_errors(dict(config)),
            key=lambda err: list(err.absolute_path),
        )
        if errors:
            raise ValueError(f"invalid agent_trajectory.v1 config: {errors[0].message}")
        serialized = str(dict(config)).lower()
        if "sequence.v1" in serialized or "logit.v1" in serialized:
            raise ValueError("config must not reference sequence.v1 or logit.v1 aliases")
        return content_sha256(dict(config))

    def plan(self, config: Mapping[str, Any]) -> AgentTrajectoryPlan:
        config_sha = self.validate(config)
        protocol = {
            "technique_id": AGENT_TRAJECTORY_TECHNIQUE_ID,
            "version": AGENT_TRAJECTORY_VERSION,
            "supervision": "oracle_or_teacher_trajectories",
            "mask_tool_results": bool(config.get("mask_tool_results", True)),
            "specialist_task": config.get("specialist_task", "generalist"),
            "byodt": "pending_review",
        }
        return AgentTrajectoryPlan(
            teacher_model_id=str(config["teacher_model_id"]),
            teacher_revision=str(config["teacher_revision"]),
            student_model_id=str(config["student_model_id"]),
            student_revision=str(config["student_revision"]),
            config_sha256=config_sha,
            protocol_sha256=content_sha256(protocol),
            objective_fields={
                "max_length": config["max_length"],
                "max_completion": config["max_completion"],
                "seed": config["seed"],
                "trajectory_corpus_sha256": config["trajectory_corpus_sha256"],
                "specialist_task": config.get("specialist_task", "generalist"),
                "mask_tool_results": bool(config.get("mask_tool_results", True)),
            },
        )


def is_registered_in_byodt_builtins() -> bool:
    """Static guard: builtins module must not mention agent_trajectory.v1.

    Avoids importing ``TechniqueRegistry`` (depends on optional backend modules).
    """
    builtins_path = (
        Path(__file__).resolve().parents[1].parent / "techniques" / "builtins.py"
    )
    tree = ast.parse(builtins_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == AGENT_TRAJECTORY_TECHNIQUE_ID:
            return True
        if isinstance(node, ast.Constant) and node.value == "agent_trajectory":
            return True
    return False
