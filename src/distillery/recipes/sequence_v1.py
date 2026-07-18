"""sequence.v1: completion-only SFT/QLoRA on prompt-response traces."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from distillery.contracts.errors import DistilleryErrorCode
from distillery.contracts.hashing import content_sha256
from distillery.contracts.recipes import RecipeId
from distillery.contracts.tasks import LabelSource
from distillery.recipes.base import (
    CompletionMask,
    JointTokenizationEvidence,
    MaterializationReport,
    MaterializedExample,
    Recipe,
    RecipeContext,
    RecipeMode,
    ResponseRecord,
    count_label_sources,
    raise_recipe_error,
    require_pinned_revision,
)

# Label used for ignored prompt/pad positions when building CE labels.
IGNORE_INDEX = -100


class SequenceV1Config(BaseModel):
    """Hyperparameters that define the sequence.v1 objective (not LoRA hardware)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recipe_id: str = RecipeId.SEQUENCE_V1.value
    mode: RecipeMode = RecipeMode.SEQUENCE_CE
    require_nonempty_response: bool = True
    require_json_object_response: bool = True
    max_completion: int = Field(default=160, ge=1)
    max_length: int = Field(default=512, ge=2)
    pad_token_id: int | None = None


def validate_response_text(
    response_text: str,
    *,
    require_nonempty: bool = True,
    require_json_object: bool = True,
) -> tuple[str, ...]:
    """Return rejection reasons (empty tuple means accepted)."""
    reasons: list[str] = []
    stripped = response_text.strip()
    if require_nonempty and not stripped:
        reasons.append("empty_response")
        return tuple(reasons)
    if require_json_object:
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            reasons.append("invalid_json")
            return tuple(reasons)
        if not isinstance(parsed, dict):
            reasons.append("response_not_json_object")
    return tuple(reasons)


def materialize_sequence_examples(
    records: Sequence[ResponseRecord],
    *,
    config: SequenceV1Config | None = None,
    task_by_example_id: dict[str, str] | None = None,
) -> MaterializationReport:
    """Validate responses and build accepted/rejected materialization lists."""
    cfg = config or SequenceV1Config()
    accepted: list[MaterializedExample] = []
    rejected: list[MaterializedExample] = []
    task_map = task_by_example_id or {}

    for record in records:
        record.assert_integrity()
        reasons = validate_response_text(
            record.selected_target_text,
            require_nonempty=cfg.require_nonempty_response,
            require_json_object=cfg.require_json_object_response,
        )
        if not record.prompt_text.strip():
            reasons = (*reasons, "empty_prompt")
        if record.completion_token_count > cfg.max_completion:
            reasons = (*reasons, "completion_token_count_exceeds_max")
        if record.total_token_count > cfg.max_length:
            reasons = (*reasons, "total_token_count_exceeds_max_length")
        example = MaterializedExample(
            example_id=record.example_id,
            task=task_map.get(record.example_id, record.task),
            difficulty=record.difficulty,
            prompt_text=record.prompt_text,
            response_text=record.selected_target_text,
            source_response_text=record.response_text,
            selected_target_text=record.selected_target_text,
            label_source=record.label_source,
            tokenization=record.tokenization,
            provenance=record.provenance_payload(),
            record_sha256=record.record_sha256,
            rejected=bool(reasons),
            rejection_reasons=reasons,
            metadata={
                "canonical_record_sha256": record.record_sha256,
            },
        )
        if reasons:
            rejected.append(example)
        else:
            accepted.append(example)

    all_examples = (*accepted, *rejected)
    return MaterializationReport(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        label_source_counts=count_label_sources(all_examples),
        recipe_id=cfg.recipe_id,
    )


def build_completion_only_mask(
    prompt_token_ids: Sequence[int],
    completion_token_ids: Sequence[int],
    *,
    max_length: int,
    max_completion: int,
    pad_token_id: int = 0,
    ignore_index: int = IGNORE_INDEX,
) -> CompletionMask:
    """
    Construct completion-only CE labels and loss mask.

    Prompt and padding positions receive zero loss weight and ``ignore_index`` labels.
    Completion tokens receive unit loss weight. Inputs must already satisfy both
    independently sealed length caps; this function never silently truncates.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, Integral):
        raise ValueError("max_length must be an integer >= 1")
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    if not completion_token_ids:
        raise ValueError("completion_token_ids must be non-empty")
    if (
        isinstance(max_completion, bool)
        or not isinstance(max_completion, Integral)
        or max_completion < 1
    ):
        raise ValueError("max_completion must be an integer >= 1")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, Integral):
        raise ValueError("pad_token_id must be an integer")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
        raise ValueError("ignore_index must be an integer")

    for name, token_ids in (
        ("prompt_token_ids", prompt_token_ids),
        ("completion_token_ids", completion_token_ids),
    ):
        for index, token_id in enumerate(token_ids):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, Integral)
                or token_id < 0
            ):
                raise ValueError(f"{name}[{index}] must be a non-negative integer")

    prompt = list(prompt_token_ids)
    completion = list(completion_token_ids)
    if len(completion) > max_completion:
        raise ValueError("completion token count exceeds max_completion")
    if len(prompt) + len(completion) > max_length:
        raise ValueError("joint token count exceeds max_length")

    input_ids = prompt + completion
    prompt_len = len(prompt)
    completion_len = len(completion)
    seq_len = len(input_ids)
    pad_len = max_length - seq_len

    attention_mask = [1] * seq_len + [0] * pad_len
    padded_ids = input_ids + [pad_token_id] * pad_len

    labels = [ignore_index] * prompt_len + list(completion) + [ignore_index] * pad_len
    loss_mask = [0.0] * prompt_len + [1.0] * completion_len + [0.0] * pad_len

    if len(labels) != max_length or len(loss_mask) != max_length:
        raise RuntimeError("internal mask length invariant violated")

    return CompletionMask(
        input_ids=tuple(padded_ids),
        attention_mask=tuple(attention_mask),
        labels=tuple(labels),
        loss_mask=tuple(loss_mask),
        prompt_token_count=prompt_len,
        completion_token_count=completion_len,
    )


def retokenize_text_pair(
    prompt_text: str,
    response_text: str,
    *,
    tokenizer_sha256: str,
    encode_with_offsets_fn: Callable[[str], Mapping[str, Any]],
) -> JointTokenizationEvidence:
    """
    Student-side retokenization of canonical text.

    The tokenizer is called exactly once on ``prompt_text + response_text`` and
    must return ``input_ids`` plus character ``offset_mapping``. A token crossing
    the text boundary is correctly treated as a completion token.
    """
    joint_text = prompt_text + response_text
    encoded = encode_with_offsets_fn(joint_text)
    if not isinstance(encoded, Mapping):
        raise ValueError("joint tokenizer output must be a mapping")
    input_ids = encoded.get("input_ids")
    offset_mapping = encoded.get("offset_mapping")
    if not isinstance(input_ids, (list, tuple)) or not isinstance(
        offset_mapping, (list, tuple)
    ):
        raise ValueError(
            "joint tokenizer output requires input_ids and offset_mapping"
        )
    completion_start = next(
        (
            index
            for index, offset in enumerate(offset_mapping)
            if isinstance(offset, (list, tuple))
            and len(offset) == 2
            and offset[1] > len(prompt_text)
        ),
        len(offset_mapping),
    )
    if completion_start >= len(input_ids):
        raise ValueError("joint tokenizer produced no completion token")
    return JointTokenizationEvidence(
        tokenizer_sha256=tokenizer_sha256,
        joint_text_sha256=content_sha256(joint_text),
        input_ids=tuple(input_ids),
        input_ids_sha256=content_sha256({"input_ids": list(input_ids)}),
        offset_mapping=tuple(tuple(offset) for offset in offset_mapping),
        offset_mapping_sha256=content_sha256(
            {"offset_mapping": [list(offset) for offset in offset_mapping]}
        ),
        completion_start_index=completion_start,
        prompt_token_count=completion_start,
        completion_token_count=len(input_ids) - completion_start,
        total_token_count=len(input_ids),
    )


def build_completion_only_mask_from_joint(
    tokenization: JointTokenizationEvidence,
    *,
    max_length: int,
    max_completion: int,
    pad_token_id: int = 0,
    ignore_index: int = IGNORE_INDEX,
) -> CompletionMask:
    """Build labels from one sealed joint tokenization and its loss boundary."""
    boundary = tokenization.completion_start_index
    return build_completion_only_mask(
        tokenization.input_ids[:boundary],
        tokenization.input_ids[boundary:],
        max_length=max_length,
        max_completion=max_completion,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
    )


class SequenceV1Recipe(Recipe):
    """Implemented sequence.v1 recipe."""

    recipe_id = RecipeId.SEQUENCE_V1

    def __init__(self, config: SequenceV1Config | None = None) -> None:
        self.config = config or SequenceV1Config()

    def validate_capabilities(self, context: RecipeContext) -> None:
        require_pinned_revision(
            context.student_revision,
            role="student",
            run_id=context.run_id,
        )
        if context.max_completion < 1 or context.max_length < 2:
            raise_recipe_error(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "sequence.v1 length bounds are invalid",
                details={
                    "max_length": context.max_length,
                    "max_completion": context.max_completion,
                },
                run_id=context.run_id,
            )
        if self.config.max_completion != context.max_completion:
            raise_recipe_error(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "sequence.v1 recipe/config completion bounds do not match",
                details={
                    "recipe_max_completion": self.config.max_completion,
                    "context_max_completion": context.max_completion,
                },
                run_id=context.run_id,
            )
        if self.config.max_length != context.max_length:
            raise_recipe_error(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "sequence.v1 recipe/config total length bounds do not match",
                details={
                    "recipe_max_length": self.config.max_length,
                    "context_max_length": context.max_length,
                },
                run_id=context.run_id,
            )

    def materialize(
        self,
        records: Sequence[ResponseRecord],
        *,
        context: RecipeContext,
    ) -> MaterializationReport:
        self.validate_capabilities(context)
        if not records:
            raise_recipe_error(
                DistilleryErrorCode.INVALID_DATASET,
                "sequence.v1 materialization received zero response records",
                run_id=context.run_id,
            )
        report = materialize_sequence_examples(records, config=self.config)
        if not report.accepted:
            raise_recipe_error(
                DistilleryErrorCode.UNSUPPORTED_LABEL_SOURCE,
                "sequence.v1 has no usable accepted responses after validation",
                details={
                    "rejected": len(report.rejected),
                    "label_source_counts": report.label_source_counts,
                },
                run_id=context.run_id,
            )
        return report

    def objective_fields(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id.value,
            "mode": self.config.mode.value,
            "objective": "ce",
            "signal": "hard_target_sequence",
        }


def example_from_envelope_response(
    *,
    example_id: str,
    task: str,
    difficulty: str,
    prompt_text: str,
    response_text: str,
    selected_target_text: str,
    tokenization: JointTokenizationEvidence,
    label_source: LabelSource = LabelSource.IMPORTED,
    provenance: dict[str, Any] | None = None,
) -> ResponseRecord:
    """Convenience builder used by tests and dry-run materialization."""
    return ResponseRecord.seal(
        example_id=example_id,
        task=task,
        difficulty=difficulty,
        prompt_text=prompt_text,
        response_text=response_text,
        selected_target_text=selected_target_text,
        label_source=label_source,
        tokenization=tokenization,
        **(provenance or {}),
    )
