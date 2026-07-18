"""Strict IID/OOD split definitions for Finance Agent."""

from __future__ import annotations

from dataclasses import dataclass

from distillery.contracts.tasks import SplitName
from distillery.finance_agent.contracts import CaseFamily, ToolName

HELD_OUT_TOOLS_OOD: tuple[ToolName, ...] = (
    ToolName.TRANSACTION_MATCHING,
    ToolName.VARIANCE_DRILL_DOWN,
)
HELD_OUT_DOMAINS_OOD: tuple[str, ...] = ("payroll",)
TRAIN_DOMAINS: tuple[str, ...] = ("travel", "software", "facilities")

_IID_CASES: tuple[CaseFamily, ...] = (
    CaseFamily.HAPPY_PATH,
    CaseFamily.WRONG_TOOL,
    CaseFamily.CORRECT_TOOL_WRONG_ARGS,
    CaseFamily.STALE_POLICY,
    CaseFamily.AMBIGUOUS_MERCHANT,
    CaseFamily.CONFLICTING_EVIDENCE,
    CaseFamily.REFUSAL_MISSING_DATA,
)
_OOD_CASES: tuple[CaseFamily, ...] = (
    CaseFamily.MULTI_STEP_RECONCILIATION,
    CaseFamily.ARITHMETIC_TRAP,
    CaseFamily.HAPPY_PATH,
    CaseFamily.WRONG_TOOL,
    CaseFamily.CORRECT_TOOL_WRONG_ARGS,
    CaseFamily.STALE_POLICY,
    CaseFamily.AMBIGUOUS_MERCHANT,
    CaseFamily.CONFLICTING_EVIDENCE,
    CaseFamily.REFUSAL_MISSING_DATA,
)


@dataclass(frozen=True)
class AgentSplitSpec:
    name: SplitName
    count: int
    token: str
    ood: bool = False
    template_variant_offset: int = 0


SMOKE_SPLITS: tuple[AgentSplitSpec, ...] = (
    AgentSplitSpec(SplitName.TRAIN, 24, "ags_tr", template_variant_offset=0),
    AgentSplitSpec(SplitName.VALIDATION, 8, "ags_va", template_variant_offset=32),
    AgentSplitSpec(SplitName.TEST, 8, "ags_te", template_variant_offset=64),
    AgentSplitSpec(SplitName.OOD_TEST, 8, "ags_ood", ood=True, template_variant_offset=96),
)

PLANNED_SPLITS: tuple[AgentSplitSpec, ...] = (
    AgentSplitSpec(SplitName.TRAIN, 1_200, "agf_tr", template_variant_offset=0),
    AgentSplitSpec(SplitName.VALIDATION, 200, "agf_va", template_variant_offset=32),
    AgentSplitSpec(SplitName.IID_TEST, 400, "agf_iid", template_variant_offset=64),
    AgentSplitSpec(
        SplitName.OOD_TEST,
        400,
        "agf_ood",
        ood=True,
        template_variant_offset=96,
    ),
)


def train_tools() -> tuple[ToolName, ...]:
    return tuple(tool for tool in ToolName if tool not in HELD_OUT_TOOLS_OOD)


def tools_for_split(spec: AgentSplitSpec) -> tuple[ToolName, ...]:
    return tuple(ToolName) if spec.ood else train_tools()


def domains_for_split(spec: AgentSplitSpec) -> tuple[str, ...]:
    return HELD_OUT_DOMAINS_OOD if spec.ood else TRAIN_DOMAINS


def case_pool_for_split(spec: AgentSplitSpec) -> tuple[CaseFamily, ...]:
    return _OOD_CASES if spec.ood else _IID_CASES


__all__ = [
    "HELD_OUT_DOMAINS_OOD",
    "HELD_OUT_TOOLS_OOD",
    "PLANNED_SPLITS",
    "SMOKE_SPLITS",
    "TRAIN_DOMAINS",
    "AgentSplitSpec",
    "case_pool_for_split",
    "domains_for_split",
    "tools_for_split",
    "train_tools",
]
