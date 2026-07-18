"""Strict tool/domain holdout and prompt leakage regressions."""

from __future__ import annotations

from distillery.contracts.tasks import SplitName
from distillery.finance_agent.generate import GeneratedAgentCorpus, _generate_one
from distillery.finance_agent.leakage import check_agent_leakage, normalize_prompt
from distillery.finance_agent.splits import (
    HELD_OUT_DOMAINS_OOD,
    HELD_OUT_TOOLS_OOD,
    PLANNED_SPLITS,
)


def test_held_out_tools_are_absent_from_all_non_ood_splits() -> None:
    held = set(HELD_OUT_TOOLS_OOD)
    for split_spec in PLANNED_SPLITS:
        generated = [
            _generate_one(corpus_seed=17, split_spec=split_spec, index=index)[0]
            for index in range(64)
        ]
        if split_spec.name is not SplitName.OOD_TEST:
            for example in generated:
                available = {definition.name for definition in example.model_input.tools}
                called = {call.tool for call in example.gold.trajectory.tool_calls()}
                assert not (available & held)
                assert not (called & held)
        else:
            assert any(
                {call.tool for call in example.gold.trajectory.tool_calls()} & held
                for example in generated
            )


def test_smoke_test_is_iid_and_ood_is_explicit(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    held = set(HELD_OUT_TOOLS_OOD)
    assert all(
        not ({call.tool for call in example.gold.trajectory.tool_calls()} & held)
        for example in smoke_corpus.by_split[SplitName.TEST]
    )
    assert all(
        example.model_input.public_world["domain"] in HELD_OUT_DOMAINS_OOD
        for example in smoke_corpus.by_split[SplitName.OOD_TEST]
    )
    assert all(
        example.model_input.public_world["accounts"][0]["code"] == "6400"
        for example in smoke_corpus.by_split[SplitName.OOD_TEST]
    )


def test_leakage_report_covers_identity_template_normalized_and_semantic_checks(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    report = smoke_corpus.leakage
    assert report.ok
    assert report.identity_overlaps == ()
    assert report.model_input_hash_overlaps == ()
    assert report.normalized_prompt_overlaps == ()
    assert report.template_family_overlaps == ()
    assert report.semantic_prompt_overlaps == ()
    assert report.gold_model_record_leaks == ()


def test_duplicate_across_splits_is_detected_adversarially(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    mutated = {split: list(items) for split, items in smoke_corpus.by_split.items()}
    mutated[SplitName.TEST][0] = mutated[SplitName.TRAIN][0]
    report = check_agent_leakage(
        mutated,
        held_out_tools=frozenset(HELD_OUT_TOOLS_OOD),
        held_out_domains=frozenset(HELD_OUT_DOMAINS_OOD),
    )
    assert not report.ok
    assert report.identity_overlaps
    assert report.model_input_hash_overlaps
    assert report.template_family_overlaps


def test_prompts_are_substantially_diverse_not_ten_repeated_strings(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    prompts = [
        "\n".join(message.text for message in example.model_input.user_messages)
        for example in smoke_corpus.examples
    ]
    normalized = {normalize_prompt(prompt) for prompt in prompts}
    assert len(set(prompts)) == 48
    assert len(normalized) >= 40
    assert len({example.provenance.template_family for example in smoke_corpus.examples}) >= 32
