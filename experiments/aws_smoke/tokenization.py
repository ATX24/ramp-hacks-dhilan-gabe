"""Pinned chat-template tokenization and sealed completion-count evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distillery.contracts.hashing import content_sha256, sha256_hex
from experiments.aws_smoke.profile import RunArm


class ChatTokenizer(Protocol):
    chat_template: str | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class TokenizedPair:
    input_ids: list[int]
    labels: list[int]
    completion_mask: list[float]
    prompt_token_count: int
    completion_token_count: int
    original_completion_token_count: int
    completion_truncated: bool


class ArmTokenizationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: RunArm
    target_source: Literal["oracle", "pre_materialized_teacher"]
    completion_token_counts: dict[str, int]
    prompt_token_counts: dict[str, int]
    total_token_counts: dict[str, int]
    record_sha256: dict[str, str]
    source_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completion_record_sha256: dict[str, str]
    original_completion_token_counts: dict[str, int]
    truncated_example_ids: tuple[str, ...] = ()
    teacher_responses_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def _validate_evidence(self) -> ArmTokenizationEvidence:
        expected_ids = set(self.completion_token_counts)
        mappings = (
            self.prompt_token_counts,
            self.total_token_counts,
            self.record_sha256,
            self.original_completion_token_counts,
            self.completion_record_sha256,
        )
        if not expected_ids or any(set(mapping) != expected_ids for mapping in mappings):
            raise ValueError("all tokenization evidence maps must cover identical ids")
        for example_id in expected_ids:
            completion = self.completion_token_counts[example_id]
            prompt = self.prompt_token_counts[example_id]
            total = self.total_token_counts[example_id]
            original = self.original_completion_token_counts[example_id]
            if completion < 1 or prompt < 1 or total != prompt + completion:
                raise ValueError("token counts must be positive and total=prompt+completion")
            if original < completion:
                raise ValueError("original completion count cannot be below capped count")
            for digest in (
                self.record_sha256[example_id],
                self.completion_record_sha256[example_id],
            ):
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise ValueError("record hashes must be 64 lowercase hex")
        if not set(self.truncated_example_ids) <= expected_ids:
            raise ValueError("truncated ids must belong to tokenization evidence")
        expected_canonical = content_sha256(
            {
                "completion_record_sha256": dict(
                    sorted(self.completion_record_sha256.items())
                )
            }
        )
        if self.canonical_records_sha256 != expected_canonical:
            raise ValueError("canonical completion-record hash is inconsistent")
        if self.target_source == "pre_materialized_teacher":
            if self.teacher_responses_sha256 != self.source_file_sha256:
                raise ValueError("teacher response source/hash evidence must match")
        elif self.teacher_responses_sha256 is not None:
            raise ValueError("oracle targets must not carry teacher response evidence")
        return self


class TokenizationEvidence(BaseModel):
    """Pre-manifest evidence produced from the exact local student tokenizer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["distillery.aws_smoke.tokenization.v1"] = (
        "distillery.aws_smoke.tokenization.v1"
    )
    student_tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    student_chat_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    student_special_token_map: dict[str, int]
    max_length: int = Field(ge=1)
    max_completion: int = Field(ge=1)
    arms: dict[RunArm, ArmTokenizationEvidence]

    def arm(self, arm: RunArm) -> ArmTokenizationEvidence:
        try:
            return self.arms[arm]
        except KeyError:
            raise ValueError(f"tokenization evidence missing arm {arm}") from None


def build_prompt_ids(
    tokenizer: ChatTokenizer,
    prompt: str,
) -> list[int]:
    _require_template(tokenizer)
    raw = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_ids = [int(value) for value in raw]
    if not prompt_ids:
        raise ValueError("pinned chat template produced an empty prompt")
    return prompt_ids


def build_chat_token_pair(
    tokenizer: ChatTokenizer,
    prompt: str,
    target: str,
    *,
    max_length: int,
    max_completion: int,
) -> TokenizedPair:
    """Apply the loaded pinned template and mask every non-completion token."""
    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    if max_completion < 1 or max_completion >= max_length:
        raise ValueError("max_completion must be in [1, max_length)")
    if not target:
        raise ValueError("target must be nonempty")

    prompt_ids = build_prompt_ids(tokenizer, prompt)
    full_raw = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )
    full_ids = [int(value) for value in full_raw]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "pinned chat template is not prefix-stable between prompt and completion"
        )
    completion_ids = full_ids[len(prompt_ids) :]
    if not completion_ids:
        raise ValueError("pinned chat template produced an empty completion")

    original_count = len(completion_ids)
    truncated = original_count > max_completion
    if truncated:
        if max_completion == 1:
            completion_ids = [completion_ids[-1]]
        else:
            completion_ids = completion_ids[: max_completion - 1] + [completion_ids[-1]]
    if len(prompt_ids) + len(completion_ids) > max_length:
        raise ValueError(
            "templated prompt plus capped completion exceeds sealed max_length; "
            "prompt truncation is forbidden because it would change the task"
        )

    input_ids = prompt_ids + completion_ids
    prompt_count = len(prompt_ids)
    completion_count = len(completion_ids)
    return TokenizedPair(
        input_ids=input_ids,
        labels=[-100] * prompt_count + completion_ids,
        completion_mask=[0.0] * prompt_count + [1.0] * completion_count,
        prompt_token_count=prompt_count,
        completion_token_count=completion_count,
        original_completion_token_count=original_count,
        completion_truncated=truncated,
    )


def load_tokenization_evidence(path: Path) -> TokenizationEvidence:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tokenization evidence must be a JSON object")
    return TokenizationEvidence.model_validate(payload)


def completion_counts_sha256(counts: dict[str, int]) -> str:
    if not counts or any(value < 1 for value in counts.values()):
        raise ValueError("completion token counts must be nonempty and positive")
    return content_sha256({"completion_token_counts": counts})


def canonical_completion_records_sha256(records: dict[str, str]) -> str:
    if not records:
        raise ValueError("completion record hashes must be nonempty")
    return content_sha256({"completion_record_sha256": dict(sorted(records.items()))})


def completion_record_sha256(
    *,
    example_id: str,
    target_text: str,
    target_source: Literal["oracle", "pre_materialized_teacher"],
) -> str:
    return content_sha256(
        {
            "example_id": example_id,
            "target_text": target_text,
            "target_source": target_source,
        }
    )


def teacher_responses_sha256(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing nonempty teacher responses: {path}")
    return sha256_hex(path.read_bytes())


def _require_template(tokenizer: ChatTokenizer) -> None:
    template = tokenizer.chat_template
    if template is None or not template.strip():
        raise ValueError("tokenizer has no pinned chat template to apply")
