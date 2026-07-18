"""Per-file, chat-template, and token-ID compatibility for each Qwen target."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distillery.contracts.hashing import content_sha256
from experiments.qwen72b_fallback.evidence import (
    REVISION_PATTERN,
    SHA256_PATTERN,
    HashBoundEvidence,
    VerificationSource,
    sha256_bytes,
)
from experiments.qwen72b_fallback.pins import (
    CHAT_TEMPLATE_SHA256,
    MODEL_ID,
    REVISION,
    SPECIAL_TOKEN_IDS,
    TOKENIZER_FILE_SHA256,
    TOKENIZER_SHA256,
    TOKENIZER_TARGETS_PATH,
)

TOKENIZER_FILENAMES = (
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class TokenizerTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model_id: str = Field(pattern=r"^Qwen/Qwen2\.5-[A-Za-z0-9.]+-Instruct$")
    revision: str = Field(pattern=REVISION_PATTERN)
    tinyfable_role: Literal[
        "nano_student",
        "core_student",
        "plus_student_candidate",
        "large_student_candidate",
    ]


class TokenizerTargetRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["distillery.qwen72b_fallback.tokenizer_targets.v1"]
    tokenizer_file_sha256: dict[str, str]
    chat_template_sha256: str = Field(pattern=SHA256_PATTERN)
    special_token_ids: dict[str, int]
    targets: tuple[TokenizerTarget, ...]

    @model_validator(mode="after")
    def _verify_registry(self) -> TokenizerTargetRegistry:
        if self.tokenizer_file_sha256 != TOKENIZER_FILE_SHA256:
            raise ValueError("target registry tokenizer files differ from Qwen72B pins")
        if self.chat_template_sha256 != CHAT_TEMPLATE_SHA256:
            raise ValueError("target registry chat template differs from Qwen72B pin")
        if self.special_token_ids != SPECIAL_TOKEN_IDS:
            raise ValueError("target registry special-token IDs differ from Qwen72B pins")
        identities = {(target.model_id, target.revision) for target in self.targets}
        if len(identities) != len(self.targets):
            raise ValueError("target registry contains duplicate model/revision pairs")
        if len(self.targets) < 3:
            raise ValueError("target registry must cover all configured TinyFable tiers")
        return self


class TokenizerPairEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.tokenizer_pair.v1"] = (
        "distillery.qwen72b_fallback.tokenizer_pair.v1"
    )
    source: Literal[VerificationSource.LIVE_AWS] = VerificationSource.LIVE_AWS
    teacher_model_id: Literal["Qwen/Qwen2.5-72B-Instruct"] = MODEL_ID
    teacher_revision: str = Field(pattern=REVISION_PATTERN)
    target_model_id: str
    target_revision: str = Field(pattern=REVISION_PATTERN)
    teacher_file_sha256: dict[str, str]
    target_file_sha256: dict[str, str]
    teacher_tokenizer_sha256: str = Field(pattern=SHA256_PATTERN)
    target_tokenizer_sha256: str = Field(pattern=SHA256_PATTERN)
    teacher_chat_template_sha256: str = Field(pattern=SHA256_PATTERN)
    target_chat_template_sha256: str = Field(pattern=SHA256_PATTERN)
    teacher_special_token_ids: dict[str, int]
    target_special_token_ids: dict[str, int]

    @model_validator(mode="after")
    def _verify_exact_pair(self) -> TokenizerPairEvidence:
        if self.teacher_revision != REVISION:
            raise ValueError("teacher tokenizer evidence uses the wrong 72B revision")
        if self.teacher_file_sha256 != self.target_file_sha256:
            raise ValueError("teacher/target tokenizer file body hashes differ")
        if self.teacher_file_sha256 != TOKENIZER_FILE_SHA256:
            raise ValueError("teacher/target tokenizer hashes differ from sealed Qwen pins")
        if self.teacher_tokenizer_sha256 != self.target_tokenizer_sha256:
            raise ValueError("teacher/target tokenizer aggregate hashes differ")
        if self.teacher_tokenizer_sha256 != TOKENIZER_SHA256:
            raise ValueError("teacher/target aggregate hash differs from sealed Qwen pin")
        if self.teacher_chat_template_sha256 != self.target_chat_template_sha256:
            raise ValueError("teacher/target chat-template hashes differ")
        if self.teacher_chat_template_sha256 != CHAT_TEMPLATE_SHA256:
            raise ValueError("teacher/target chat-template differs from sealed Qwen pin")
        if self.teacher_special_token_ids != self.target_special_token_ids:
            raise ValueError("teacher/target special-token IDs differ")
        if self.teacher_special_token_ids != SPECIAL_TOKEN_IDS:
            raise ValueError("teacher/target special-token IDs differ from sealed Qwen pins")
        return self


class TokenizerCompatibilityEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.tokenizer_compat.v1"] = (
        "distillery.qwen72b_fallback.tokenizer_compat.v1"
    )
    source: Literal[VerificationSource.LIVE_AWS] = VerificationSource.LIVE_AWS
    registry_bytes_sha256: str = Field(pattern=SHA256_PATTERN)
    pairs: tuple[TokenizerPairEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _all_registered_targets_present(self) -> TokenizerCompatibilityEvidence:
        registry = load_target_registry()
        expected = {(target.model_id, target.revision) for target in registry.targets}
        actual = {(pair.target_model_id, pair.target_revision) for pair in self.pairs}
        if actual != expected:
            raise ValueError("tokenizer evidence does not cover every registered target pair")
        return self


@lru_cache(maxsize=1)
def load_target_registry() -> TokenizerTargetRegistry:
    return TokenizerTargetRegistry.model_validate_json(TOKENIZER_TARGETS_PATH.read_bytes())


def _file_hashes(bodies: dict[str, bytes]) -> dict[str, str]:
    if set(bodies) != set(TOKENIZER_FILENAMES):
        raise ValueError(
            "tokenizer body set mismatch: "
            f"expected={list(TOKENIZER_FILENAMES)} actual={sorted(bodies)}"
        )
    return {name: sha256_bytes(bodies[name]) for name in TOKENIZER_FILENAMES}


def _chat_template_hash(tokenizer_config_bytes: bytes) -> str:
    config = json.loads(tokenizer_config_bytes)
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer_config.json lacks a non-empty chat_template")
    return sha256_bytes(template.encode("utf-8"))


def _special_token_ids(tokenizer_json_bytes: bytes) -> dict[str, int]:
    payload = json.loads(tokenizer_json_bytes)
    added = payload.get("added_tokens")
    if not isinstance(added, list):
        raise ValueError("tokenizer.json lacks added_tokens")
    actual: dict[str, int] = {}
    for entry in added:
        if not isinstance(entry, dict):
            raise ValueError("tokenizer.json added_tokens entry is not an object")
        content = entry.get("content")
        token_id = entry.get("id")
        if content in SPECIAL_TOKEN_IDS:
            if not isinstance(token_id, int):
                raise ValueError(f"token {content!r} lacks an integer id")
            actual[str(content)] = token_id
    if actual != SPECIAL_TOKEN_IDS:
        raise ValueError("tokenizer.json special token IDs differ from sealed Qwen pins")
    return actual


def verify_tokenizer_pair(
    *,
    target: TokenizerTarget,
    teacher_bodies: dict[str, bytes],
    target_bodies: dict[str, bytes],
) -> TokenizerPairEvidence:
    teacher_hashes = _file_hashes(teacher_bodies)
    target_hashes = _file_hashes(target_bodies)
    teacher_chat = _chat_template_hash(teacher_bodies["tokenizer_config.json"])
    target_chat = _chat_template_hash(target_bodies["tokenizer_config.json"])
    teacher_ids = _special_token_ids(teacher_bodies["tokenizer.json"])
    target_ids = _special_token_ids(target_bodies["tokenizer.json"])
    teacher_aggregate = content_sha256({"tokenizer_files": teacher_hashes})
    target_aggregate = content_sha256({"tokenizer_files": target_hashes})
    return TokenizerPairEvidence.seal(
        teacher_revision=REVISION,
        target_model_id=target.model_id,
        target_revision=target.revision,
        teacher_file_sha256=teacher_hashes,
        target_file_sha256=target_hashes,
        teacher_tokenizer_sha256=teacher_aggregate,
        target_tokenizer_sha256=target_aggregate,
        teacher_chat_template_sha256=teacher_chat,
        target_chat_template_sha256=target_chat,
        teacher_special_token_ids=teacher_ids,
        target_special_token_ids=target_ids,
    )


def seal_compatibility(
    pairs: tuple[TokenizerPairEvidence, ...],
) -> TokenizerCompatibilityEvidence:
    return TokenizerCompatibilityEvidence.seal(
        registry_bytes_sha256=sha256_bytes(TOKENIZER_TARGETS_PATH.read_bytes()),
        pairs=pairs,
    )
