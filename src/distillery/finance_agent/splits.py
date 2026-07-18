"""Held-out tool and domain splits for Finance Agent corpora."""

from __future__ import annotations

from dataclasses import dataclass

from distillery.contracts.tasks import SplitName
from distillery.finance_agent.contracts import ToolName

HELD_OUT_TOOLS_OOD: tuple[ToolName, ...] = (
    ToolName.VARIANCE_DRILL_DOWN,
    ToolName.TRANSACTION_MATCHING,
)
HELD_OUT_DOMAINS_OOD: tuple[str, ...] = ("payroll",)


@dataclass(frozen=True)
class AgentSplitSpec:
    name: SplitName
    count: int
    token: str
    hold_out_tools: bool = False
    hold_out_domains: bool = False


SMOKE_SPLITS: tuple[AgentSplitSpec, ...] = (
    AgentSplitSpec(SplitName.TRAIN, 24, "ags_tr"),
    AgentSplitSpec(SplitName.VALIDATION, 8, "ags_va"),
    AgentSplitSpec(SplitName.TEST, 16, "ags_te"),
)

PLANNED_SPLITS: tuple[AgentSplitSpec, ...] = (
    AgentSplitSpec(SplitName.TRAIN, 1_200, "agf_tr"),
    AgentSplitSpec(SplitName.VALIDATION, 200, "agf_va"),
    AgentSplitSpec(SplitName.IID_TEST, 400, "agf_iid"),
    AgentSplitSpec(SplitName.OOD_TEST, 400, "agf_ood", hold_out_tools=True, hold_out_domains=True),
)


def train_tools() -> tuple[ToolName, ...]:
    return tuple(tool for tool in ToolName if tool not in HELD_OUT_TOOLS_OOD)


def tools_for_split(spec: AgentSplitSpec) -> tuple[ToolName, ...]:
    if spec.hold_out_tools:
        return tuple(ToolName)
    return train_tools()


def domains_for_split(spec: AgentSplitSpec) -> tuple[str, ...] | None:
    """Return None when all domains are allowed; else the allowed training domains."""
    if spec.hold_out_domains:
        return None
    return ("travel", "software", "facilities")
