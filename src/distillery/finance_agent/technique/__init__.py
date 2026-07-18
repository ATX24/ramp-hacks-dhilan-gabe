"""Isolated agent_trajectory.v1 plan adapter (not registered in BYODT builtins)."""

from __future__ import annotations

from distillery.finance_agent.technique.adapter import (
    AGENT_TRAJECTORY_TECHNIQUE_ID,
    AgentTrajectoryPlan,
    AgentTrajectoryPlanAdapter,
    assert_not_sequence_or_logit_alias,
)
from distillery.finance_agent.technique.tokenization import (
    IGNORE_INDEX,
    TRAJECTORY_RENDER_TEMPLATE_SHA256,
    AgentTrajectoryBatch,
    AgentTrajectoryCollatorConfig,
    collate_agent_trajectories,
    tokenize_agent_trajectory,
)

__all__ = [
    "AGENT_TRAJECTORY_TECHNIQUE_ID",
    "IGNORE_INDEX",
    "TRAJECTORY_RENDER_TEMPLATE_SHA256",
    "AgentTrajectoryBatch",
    "AgentTrajectoryCollatorConfig",
    "AgentTrajectoryPlan",
    "AgentTrajectoryPlanAdapter",
    "assert_not_sequence_or_logit_alias",
    "collate_agent_trajectories",
    "tokenize_agent_trajectory",
]
