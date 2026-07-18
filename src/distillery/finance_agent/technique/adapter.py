"""Isolated plan adapter for the role-masked agent_trajectory.v1 objective."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import StrictStr

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import Sha256Hex, content_sha256
from distillery.finance_agent.contracts import TECHNIQUE_ID_AGENT_TRAJECTORY
from distillery.finance_agent.proof import FINANCE_AGENT_METRICS
from distillery.finance_agent.technique.tokenization import (
    IGNORE_INDEX,
    TRAJECTORY_RENDER_TEMPLATE_SHA256,
    AgentTrajectoryCollatorConfig,
)

AGENT_TRAJECTORY_TECHNIQUE_ID: Literal["agent_trajectory.v1"] = TECHNIQUE_ID_AGENT_TRAJECTORY
AGENT_TRAJECTORY_VERSION: Literal["1.1.0"] = "1.1.0"

_FORBIDDEN_ALIASES = frozenset({"sequence.v1", "logit.v1", "sequence", "logit"})
_FORBIDDEN_CLAIM_KEYS = frozenset(
    {
        "teacher_model_id",
        "teacher_revision",
        "teacher_signal",
        "teacher_rollout",
        "specialist_task",
    }
)

CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "max_length",
        "max_supervised_tokens",
        "pad_token_id",
        "seed",
        "corpus_seed",
        "trajectory_corpus_sha256",
        "corpus_order_sha256",
        "system_prompt_set_sha256",
        "tool_schema_set_sha256",
        "trajectory_render_template_sha256",
        "mask_tool_results",
        "label_source",
    ],
    "properties": {
        "max_length": {"type": "integer", "minimum": 2, "maximum": 131_072},
        "max_supervised_tokens": {
            "type": "integer",
            "minimum": 1,
            "maximum": 131_072,
        },
        "pad_token_id": {"type": "integer", "minimum": 0},
        "seed": {"type": "integer", "minimum": 0},
        "corpus_seed": {"type": "integer", "minimum": 0},
        "trajectory_corpus_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "corpus_order_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "system_prompt_set_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "tool_schema_set_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "trajectory_render_template_sha256": {
            "const": TRAJECTORY_RENDER_TEMPLATE_SHA256,
        },
        "mask_tool_results": {"const": True},
        "label_source": {"const": "oracle"},
        "student_model_id": {"type": ["string", "null"], "minLength": 1},
        "student_revision": {
            "type": ["string", "null"],
            "pattern": "^[0-9a-f]{40}$",
        },
        "tokenizer_sha256": {
            "type": ["string", "null"],
            "pattern": "^[0-9a-f]{64}$",
        },
        "chat_template_sha256": {
            "type": ["string", "null"],
            "pattern": "^[0-9a-f]{64}$",
        },
        "license_status": {"const": "unknown"},
        "cost_status": {"const": "unknown"},
    },
}


class AgentTrajectoryPlan(FrozenModel):
    """Plan-only objective contract; not executable through BYODT builtins."""

    schema_version: Literal["finance_agent.technique_plan.v2"] = "finance_agent.technique_plan.v2"
    technique_id: Literal["agent_trajectory.v1"] = AGENT_TRAJECTORY_TECHNIQUE_ID
    version: Literal["1.1.0"] = AGENT_TRAJECTORY_VERSION
    display_name: StrictStr = "Role-masked agent trajectory objective"
    supervision_source: Literal["oracle_trajectory_labels"] = "oracle_trajectory_labels"
    student_target: Literal["tinyfable_generalist"] = "tinyfable_generalist"
    objective: Literal["role_masked_trajectory_ce"] = "role_masked_trajectory_ce"
    config_sha256: Sha256Hex
    objective_protocol_sha256: Sha256Hex
    metrics: tuple[StrictStr, ...] = FINANCE_AGENT_METRICS
    collator_config: AgentTrajectoryCollatorConfig
    objective_fields: FrozenJsonObject
    byodt_integration: Literal["pending_review"] = "pending_review"
    training_status: Literal["not_materialized"] = "not_materialized"
    training_ready: Literal[False] = False
    readiness_blockers: tuple[StrictStr, ...]
    notes: StrictStr = (
        "Current labels are deterministic oracle trajectories. No teacher rollout, "
        "teacher model, specialist routing, logits, training artifact, or measured "
        "economics is represented by this plan."
    )

    def plan_hash(self) -> str:
        return content_sha256(self.model_dump(mode="json"))


def assert_not_sequence_or_logit_alias(technique_id: str) -> None:
    if technique_id in _FORBIDDEN_ALIASES:
        raise ValueError(f"{technique_id!r} is forbidden; use {AGENT_TRAJECTORY_TECHNIQUE_ID}")
    if technique_id != AGENT_TRAJECTORY_TECHNIQUE_ID:
        raise ValueError(f"expected {AGENT_TRAJECTORY_TECHNIQUE_ID!r}, got {technique_id!r}")


class AgentTrajectoryPlanAdapter:
    """Isolated adapter. It validates and describes the objective but cannot launch it."""

    technique_id = AGENT_TRAJECTORY_TECHNIQUE_ID
    version = AGENT_TRAJECTORY_VERSION

    def validate(self, config: Mapping[str, Any]) -> str:
        from jsonschema import Draft202012Validator

        assert_not_sequence_or_logit_alias(self.technique_id)
        raw = dict(config)
        forbidden = sorted(_FORBIDDEN_CLAIM_KEYS & raw.keys())
        if forbidden:
            raise ValueError(f"unsupported unproven claim fields: {forbidden}")
        errors = sorted(
            Draft202012Validator(CONFIG_SCHEMA).iter_errors(raw),
            key=lambda error: (list(error.absolute_path), error.message),
        )
        if errors:
            raise ValueError(f"invalid agent_trajectory.v1 config: {errors[0].message}")
        serialized = str(raw).lower()
        if "sequence.v1" in serialized or "logit.v1" in serialized:
            raise ValueError("config must not reference sequence.v1 or logit.v1")
        if raw["max_supervised_tokens"] > raw["max_length"]:
            raise ValueError("max_supervised_tokens cannot exceed max_length")
        return content_sha256(raw)

    def plan(self, config: Mapping[str, Any]) -> AgentTrajectoryPlan:
        raw = dict(config)
        config_sha256 = self.validate(raw)
        collator = AgentTrajectoryCollatorConfig(
            max_length=raw["max_length"],
            max_supervised_tokens=raw["max_supervised_tokens"],
            pad_token_id=raw["pad_token_id"],
            ignore_index=IGNORE_INDEX,
            mask_tool_results=raw["mask_tool_results"],
        )
        objective_protocol = {
            "technique_id": AGENT_TRAJECTORY_TECHNIQUE_ID,
            "version": AGENT_TRAJECTORY_VERSION,
            "objective": "role_masked_trajectory_ce",
            "supervised": [
                "assistant_message",
                "assistant_tool_call",
                "assistant_final_answer",
            ],
            "ignored": ["system", "user", "tool_result", "padding"],
            "ignore_index": IGNORE_INDEX,
            "render_template_sha256": TRAJECTORY_RENDER_TEMPLATE_SHA256,
            "mask_tool_results": True,
            "label_source": "oracle",
        }
        blockers = [
            "byodt_review_pending",
            "training_path_not_wired",
            "training_artifact_missing",
            "measured_economics_missing",
        ]
        for key in (
            "student_model_id",
            "student_revision",
            "tokenizer_sha256",
            "chat_template_sha256",
        ):
            if raw.get(key) is None:
                blockers.append(f"missing_{key}")
        blockers.append("license_disposition_unknown")
        return AgentTrajectoryPlan(
            config_sha256=config_sha256,
            objective_protocol_sha256=content_sha256(objective_protocol),
            collator_config=collator,
            objective_fields={
                "seed": raw["seed"],
                "corpus_seed": raw["corpus_seed"],
                "trajectory_corpus_sha256": raw["trajectory_corpus_sha256"],
                "corpus_order_sha256": raw["corpus_order_sha256"],
                "system_prompt_set_sha256": raw["system_prompt_set_sha256"],
                "tool_schema_set_sha256": raw["tool_schema_set_sha256"],
                "trajectory_render_template_sha256": raw["trajectory_render_template_sha256"],
                "student_model_id": raw.get("student_model_id"),
                "student_revision": raw.get("student_revision"),
                "tokenizer_sha256": raw.get("tokenizer_sha256"),
                "chat_template_sha256": raw.get("chat_template_sha256"),
                "label_source": "oracle",
                "license_status": raw.get("license_status", "unknown"),
                "cost_status": raw.get("cost_status", "unknown"),
            },
            readiness_blockers=tuple(blockers),
        )


def is_registered_in_byodt_builtins() -> bool:
    builtins_path = Path(__file__).resolve().parents[1].parent / "techniques" / "builtins.py"
    tree = ast.parse(builtins_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in {
            AGENT_TRAJECTORY_TECHNIQUE_ID,
            "agent_trajectory",
        }:
            return True
    return False


__all__ = [
    "AGENT_TRAJECTORY_TECHNIQUE_ID",
    "AGENT_TRAJECTORY_VERSION",
    "CONFIG_SCHEMA",
    "AgentTrajectoryPlan",
    "AgentTrajectoryPlanAdapter",
    "assert_not_sequence_or_logit_alias",
    "is_registered_in_byodt_builtins",
]
