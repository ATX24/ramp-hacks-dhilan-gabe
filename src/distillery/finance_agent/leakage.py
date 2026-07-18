"""Leakage checks across Finance Agent splits."""

from __future__ import annotations

from dataclasses import dataclass

from distillery.contracts.tasks import SplitName
from distillery.finance_agent.contracts import AgentEpisodeEnvelope


@dataclass(frozen=True)
class AgentLeakageReport:
    ok: bool
    overlapping_worlds: tuple[str, ...]
    overlapping_groups: tuple[str, ...]
    overlapping_example_ids: tuple[str, ...]
    train_tool_leak_into_holdout_definition: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "overlapping_worlds": list(self.overlapping_worlds),
            "overlapping_groups": list(self.overlapping_groups),
            "overlapping_example_ids": list(self.overlapping_example_ids),
            "train_tool_leak_into_holdout_definition": list(
                self.train_tool_leak_into_holdout_definition
            ),
        }


def check_agent_leakage(
    by_split: dict[SplitName, list[AgentEpisodeEnvelope]],
    *,
    held_out_tools: frozenset[str],
) -> AgentLeakageReport:
    train = by_split.get(SplitName.TRAIN, [])
    val = by_split.get(SplitName.VALIDATION, [])
    tests = (
        by_split.get(SplitName.TEST, [])
        + by_split.get(SplitName.IID_TEST, [])
        + by_split.get(SplitName.OOD_TEST, [])
    )
    train_worlds = {ex.world_id for ex in train}
    train_groups = {ex.group_id for ex in train}
    train_ids = {ex.example_id for ex in train}
    other = val + tests
    overlapping_worlds = tuple(sorted(train_worlds & {ex.world_id for ex in other}))
    overlapping_groups = tuple(sorted(train_groups & {ex.group_id for ex in other}))
    overlapping_ids = tuple(sorted(train_ids & {ex.example_id for ex in other}))

    leaked_tools: list[str] = []
    for ex in train:
        for turn in ex.trajectory.turns:
            if turn.tool_call and turn.tool_call.tool.value in held_out_tools:
                leaked_tools.append(f"{ex.example_id}:{turn.tool_call.tool.value}")

    ok = not (overlapping_worlds or overlapping_groups or overlapping_ids or leaked_tools)
    return AgentLeakageReport(
        ok=ok,
        overlapping_worlds=overlapping_worlds,
        overlapping_groups=overlapping_groups,
        overlapping_example_ids=overlapping_ids,
        train_tool_leak_into_holdout_definition=tuple(sorted(leaked_tools)),
    )
