"""Exact role-aware token/label mask tests for agent_trajectory.v1."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distillery.finance_agent.contracts import CaseFamily, TurnRole
from distillery.finance_agent.generate import GeneratedAgentCorpus
from distillery.finance_agent.technique.tokenization import (
    IGNORE_INDEX,
    AgentTrajectoryCollatorConfig,
    collate_agent_trajectories,
    render_role_segments,
    tokenize_agent_trajectory,
)


class CharacterTokenizer:
    """Deterministic tokenizer whose exact segment masks are inspectable."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) + 1 for character in text]


def test_exact_role_masks_supervise_only_assistant_segments(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    example = next(
        item for item in smoke_corpus.examples if item.case_family is CaseFamily.AMBIGUOUS_MERCHANT
    )
    tokenizer = CharacterTokenizer()
    segments = render_role_segments(example)
    unpadded_count = sum(len(tokenizer.encode(segment.text)) for segment in segments)
    config = AgentTrajectoryCollatorConfig(
        max_length=unpadded_count + 7,
        max_supervised_tokens=unpadded_count,
        pad_token_id=0,
        mask_tool_results=True,
    )
    tokenized = tokenize_agent_trajectory(
        example,
        tokenizer=tokenizer,
        config=config,
    )

    expected_ids: list[int] = []
    expected_labels: list[int] = []
    expected_loss: list[int] = []
    for segment in segments:
        ids = tokenizer.encode(segment.text)
        expected_ids.extend(ids)
        if segment.role is TurnRole.ASSISTANT:
            expected_labels.extend(ids)
            expected_loss.extend([1] * len(ids))
        else:
            expected_labels.extend([IGNORE_INDEX] * len(ids))
            expected_loss.extend([0] * len(ids))
    expected_ids.extend([0] * 7)
    expected_labels.extend([IGNORE_INDEX] * 7)
    expected_loss.extend([0] * 7)

    assert list(tokenized.input_ids) == expected_ids
    assert list(tokenized.labels) == expected_labels
    assert list(tokenized.loss_mask) == expected_loss
    assert list(tokenized.attention_mask) == [1] * unpadded_count + [0] * 7

    by_kind = {span.kind: span for span in tokenized.spans}
    for kind in ("system", "user", "tool_result"):
        span = by_kind[kind]
        assert set(tokenized.labels[span.start : span.end]) == {IGNORE_INDEX}
        assert set(tokenized.loss_mask[span.start : span.end]) == {0}
    for kind in (
        "assistant_message",
        "assistant_tool_call",
        "assistant_final_answer",
    ):
        span = by_kind[kind]
        assert tokenized.labels[span.start : span.end] == tokenized.input_ids[span.start : span.end]
        assert set(tokenized.loss_mask[span.start : span.end]) == {1}


def test_mask_tool_results_false_is_invalid_not_inert() -> None:
    with pytest.raises(ValidationError):
        AgentTrajectoryCollatorConfig(
            max_length=100,
            max_supervised_tokens=50,
            pad_token_id=0,
            mask_tool_results=False,
        )


def test_collator_batches_equal_length_rows_and_preserves_masks(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    examples = smoke_corpus.examples[:2]
    config = AgentTrajectoryCollatorConfig(
        max_length=20_000,
        max_supervised_tokens=10_000,
        pad_token_id=0,
    )
    batch = collate_agent_trajectories(
        examples,
        tokenizer=CharacterTokenizer(),
        config=config,
    )
    assert batch.example_ids == tuple(example.example_id for example in examples)
    assert all(len(row) == config.max_length for row in batch.input_ids)
    assert all(
        label == IGNORE_INDEX
        for tokenization in batch.tokenizations
        for span in tokenization.spans
        if span.role in {TurnRole.SYSTEM, TurnRole.USER, TurnRole.TOOL}
        for label in tokenization.labels[span.start : span.end]
    )


def test_tokenization_fails_loud_instead_of_truncating(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    example = smoke_corpus.examples[0]
    config = AgentTrajectoryCollatorConfig(
        max_length=10,
        max_supervised_tokens=10,
        pad_token_id=0,
    )
    with pytest.raises(ValueError, match="exceeds"):
        tokenize_agent_trajectory(
            example,
            tokenizer=CharacterTokenizer(),
            config=config,
        )
