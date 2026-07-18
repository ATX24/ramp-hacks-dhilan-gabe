"""Role-aware tokenization and collation for agent_trajectory.v1."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import Field, StrictBool, StrictInt, model_validator

from distillery.contracts.base import FrozenModel
from distillery.contracts.hashing import content_sha256
from distillery.finance_agent.contracts import (
    AgentEpisodeEnvelope,
    TurnRole,
)

IGNORE_INDEX = -100
_ROLE_PREFIX = {
    TurnRole.SYSTEM: "<|system|>\n",
    TurnRole.USER: "<|user|>\n",
    TurnRole.ASSISTANT: "<|assistant|>\n",
    TurnRole.TOOL: "<|tool|>\n",
}
_SEGMENT_SUFFIX = "\n<|end|>\n"

TRAJECTORY_RENDER_TEMPLATE = {
    "schema_version": "finance_agent.trajectory_render.v1",
    "role_prefix": {role.value: prefix for role, prefix in _ROLE_PREFIX.items()},
    "segment_suffix": _SEGMENT_SUFFIX,
    "assistant_tool_call": {"json_key": "tool_call", "canonical": "sorted_compact_json"},
    "assistant_final": {"json_key": "final_answer", "canonical": "sorted_compact_json"},
    "tool_result": {"json_key": "tool_result", "canonical": "sorted_compact_json"},
    "supervised_roles": ["assistant"],
    "masked_roles": ["system", "user", "tool"],
}
TRAJECTORY_RENDER_TEMPLATE_SHA256 = content_sha256(TRAJECTORY_RENDER_TEMPLATE)


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]:
        """Encode one already-framed role segment."""


class AgentTrajectoryCollatorConfig(FrozenModel):
    schema_version: Literal["finance_agent.collator.v1"] = "finance_agent.collator.v1"
    max_length: StrictInt = Field(ge=2, le=131_072)
    max_supervised_tokens: StrictInt = Field(ge=1, le=131_072)
    pad_token_id: StrictInt = Field(ge=0)
    ignore_index: Literal[IGNORE_INDEX] = IGNORE_INDEX
    mask_system: Literal[True] = True
    mask_user: Literal[True] = True
    mask_tool_results: Literal[True] = True
    supervise_assistant_tool_calls: Literal[True] = True
    supervise_assistant_messages: Literal[True] = True
    supervise_assistant_final_answers: Literal[True] = True


class RoleSegment(FrozenModel):
    role: TurnRole
    kind: Literal[
        "system",
        "user",
        "assistant_message",
        "assistant_tool_call",
        "tool_result",
        "assistant_final_answer",
    ]
    text: str = Field(min_length=1)
    supervised: StrictBool


class TokenSpan(FrozenModel):
    role: TurnRole
    kind: str
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(ge=1)
    supervised: StrictBool

    @model_validator(mode="after")
    def _ordered(self) -> TokenSpan:
        if self.end <= self.start:
            raise ValueError("token span end must be greater than start")
        return self


class TrajectoryTokenization(FrozenModel):
    schema_version: Literal["finance_agent.tokenization.v1"] = "finance_agent.tokenization.v1"
    example_id: str
    input_ids: tuple[StrictInt, ...]
    attention_mask: tuple[Literal[0, 1], ...]
    labels: tuple[StrictInt, ...]
    loss_mask: tuple[Literal[0, 1], ...]
    spans: tuple[TokenSpan, ...]
    supervised_token_count: StrictInt = Field(ge=1)
    prompt_token_count: StrictInt = Field(ge=1)
    render_template_sha256: Literal[TRAJECTORY_RENDER_TEMPLATE_SHA256] = (
        TRAJECTORY_RENDER_TEMPLATE_SHA256
    )

    @model_validator(mode="after")
    def _shape(self) -> TrajectoryTokenization:
        lengths = {
            len(self.input_ids),
            len(self.attention_mask),
            len(self.labels),
            len(self.loss_mask),
        }
        if len(lengths) != 1:
            raise ValueError("input_ids/attention/labels/loss_mask lengths must match")
        if sum(self.loss_mask) != self.supervised_token_count:
            raise ValueError("supervised_token_count must equal loss_mask sum")
        for token_id, label, mask, attention in zip(
            self.input_ids,
            self.labels,
            self.loss_mask,
            self.attention_mask,
            strict=True,
        ):
            if attention == 0:
                if label != IGNORE_INDEX or mask != 0:
                    raise ValueError("padding must be ignored")
            elif mask == 1 and label != token_id:
                raise ValueError("supervised labels must equal input token ids")
            elif mask == 0 and label != IGNORE_INDEX:
                raise ValueError("masked labels must use ignore_index")
        return self


class AgentTrajectoryBatch(FrozenModel):
    schema_version: Literal["finance_agent.batch.v1"] = "finance_agent.batch.v1"
    example_ids: tuple[str, ...]
    input_ids: tuple[tuple[StrictInt, ...], ...]
    attention_mask: tuple[tuple[Literal[0, 1], ...], ...]
    labels: tuple[tuple[StrictInt, ...], ...]
    loss_mask: tuple[tuple[Literal[0, 1], ...], ...]
    tokenizations: tuple[TrajectoryTokenization, ...]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _frame(role: TurnRole, payload: str) -> str:
    return f"{_ROLE_PREFIX[role]}{payload}{_SEGMENT_SUFFIX}"


def render_role_segments(example: AgentEpisodeEnvelope) -> tuple[RoleSegment, ...]:
    """Render exact role segments before tokenizer-specific encoding."""
    segments = [
        RoleSegment(
            role=TurnRole.SYSTEM,
            kind="system",
            text=_frame(TurnRole.SYSTEM, example.model_input.system_prompt),
            supervised=False,
        )
    ]
    for turn in example.gold.trajectory.turns:
        if turn.role is TurnRole.USER:
            assert turn.text is not None
            segments.append(
                RoleSegment(
                    role=TurnRole.USER,
                    kind="user",
                    text=_frame(TurnRole.USER, turn.text),
                    supervised=False,
                )
            )
        elif turn.role is TurnRole.TOOL:
            assert turn.tool_result is not None
            payload = _canonical_json({"tool_result": turn.tool_result.model_dump(mode="json")})
            segments.append(
                RoleSegment(
                    role=TurnRole.TOOL,
                    kind="tool_result",
                    text=_frame(TurnRole.TOOL, payload),
                    supervised=False,
                )
            )
        elif turn.tool_call is not None:
            payload = _canonical_json({"tool_call": turn.tool_call.model_dump(mode="json")})
            segments.append(
                RoleSegment(
                    role=TurnRole.ASSISTANT,
                    kind="assistant_tool_call",
                    text=_frame(TurnRole.ASSISTANT, payload),
                    supervised=True,
                )
            )
        elif turn.final_answer is not None:
            payload = _canonical_json({"final_answer": turn.final_answer.model_dump(mode="json")})
            segments.append(
                RoleSegment(
                    role=TurnRole.ASSISTANT,
                    kind="assistant_final_answer",
                    text=_frame(TurnRole.ASSISTANT, payload),
                    supervised=True,
                )
            )
        else:
            assert turn.text is not None
            segments.append(
                RoleSegment(
                    role=TurnRole.ASSISTANT,
                    kind="assistant_message",
                    text=_frame(TurnRole.ASSISTANT, turn.text),
                    supervised=True,
                )
            )
    return tuple(segments)


def tokenize_agent_trajectory(
    example: AgentEpisodeEnvelope,
    *,
    tokenizer: TokenizerLike,
    config: AgentTrajectoryCollatorConfig,
) -> TrajectoryTokenization:
    """Tokenize by role and apply CE only to assistant-authored segments."""
    if config.mask_tool_results is not True:
        raise ValueError("agent_trajectory.v1 requires tool-result masking")
    input_ids: list[int] = []
    labels: list[int] = []
    loss_mask: list[int] = []
    spans: list[TokenSpan] = []
    for segment in render_role_segments(example):
        raw_ids = tokenizer.encode(segment.text, add_special_tokens=False)
        token_ids = [int(token_id) for token_id in raw_ids]
        if not token_ids:
            raise ValueError(f"tokenizer emitted no ids for {segment.kind}")
        if any(token_id < 0 for token_id in token_ids):
            raise ValueError("tokenizer emitted a negative token id")
        start = len(input_ids)
        input_ids.extend(token_ids)
        if segment.supervised:
            labels.extend(token_ids)
            loss_mask.extend([1] * len(token_ids))
        else:
            labels.extend([config.ignore_index] * len(token_ids))
            loss_mask.extend([0] * len(token_ids))
        spans.append(
            TokenSpan(
                role=segment.role,
                kind=segment.kind,
                start=start,
                end=len(input_ids),
                supervised=segment.supervised,
            )
        )
    supervised_count = sum(loss_mask)
    if supervised_count == 0:
        raise ValueError("trajectory has no assistant supervision tokens")
    if supervised_count > config.max_supervised_tokens:
        raise ValueError("assistant supervision exceeds max_supervised_tokens")
    if len(input_ids) > config.max_length:
        raise ValueError("trajectory token count exceeds max_length; truncation is forbidden")
    prompt_count = len(input_ids) - supervised_count
    pad_count = config.max_length - len(input_ids)
    attention_mask = [1] * len(input_ids) + [0] * pad_count
    input_ids.extend([config.pad_token_id] * pad_count)
    labels.extend([config.ignore_index] * pad_count)
    loss_mask.extend([0] * pad_count)
    return TrajectoryTokenization(
        example_id=example.example_id,
        input_ids=tuple(input_ids),
        attention_mask=tuple(attention_mask),
        labels=tuple(labels),
        loss_mask=tuple(loss_mask),
        spans=tuple(spans),
        supervised_token_count=supervised_count,
        prompt_token_count=prompt_count,
    )


def collate_agent_trajectories(
    examples: Sequence[AgentEpisodeEnvelope],
    *,
    tokenizer: TokenizerLike,
    config: AgentTrajectoryCollatorConfig,
) -> AgentTrajectoryBatch:
    if not examples:
        raise ValueError("cannot collate an empty trajectory batch")
    tokenizations = tuple(
        tokenize_agent_trajectory(example, tokenizer=tokenizer, config=config)
        for example in examples
    )
    return AgentTrajectoryBatch(
        example_ids=tuple(item.example_id for item in tokenizations),
        input_ids=tuple(item.input_ids for item in tokenizations),
        attention_mask=tuple(item.attention_mask for item in tokenizations),
        labels=tuple(item.labels for item in tokenizations),
        loss_mask=tuple(item.loss_mask for item in tokenizations),
        tokenizations=tokenizations,
    )


__all__ = [
    "IGNORE_INDEX",
    "TRAJECTORY_RENDER_TEMPLATE",
    "TRAJECTORY_RENDER_TEMPLATE_SHA256",
    "AgentTrajectoryBatch",
    "AgentTrajectoryCollatorConfig",
    "RoleSegment",
    "TokenSpan",
    "TokenizerLike",
    "TrajectoryTokenization",
    "collate_agent_trajectories",
    "render_role_segments",
    "tokenize_agent_trajectory",
]
