"""Held-out tool/domain splits and leakage."""

from __future__ import annotations

from distillery.contracts.tasks import SplitName
from distillery.finance_agent.generate import generate_agent_corpus
from distillery.finance_agent.splits import HELD_OUT_DOMAINS_OOD, HELD_OUT_TOOLS_OOD


def test_train_does_not_use_held_out_tools() -> None:
    corpus = generate_agent_corpus("smoke")
    held = {tool.value for tool in HELD_OUT_TOOLS_OOD}
    for example in corpus.by_split[SplitName.TRAIN]:
        for turn in example.trajectory.turns:
            if turn.tool_call is not None:
                assert turn.tool_call.tool.value not in held


def test_leakage_report_ok() -> None:
    corpus = generate_agent_corpus("smoke")
    assert corpus.leakage.ok
    assert corpus.leakage.overlapping_worlds == ()
    assert corpus.leakage.overlapping_groups == ()


def test_planned_ood_split_metadata() -> None:
    from distillery.finance_agent.splits import PLANNED_SPLITS

    ood = next(split for split in PLANNED_SPLITS if split.name is SplitName.OOD_TEST)
    assert ood.hold_out_tools is True
    assert ood.hold_out_domains is True
    assert "payroll" in HELD_OUT_DOMAINS_OOD
