"""Offline-only snapshots, tokenizer hashes, templates, and completion caps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.aws_smoke.model_evidence import (
    assert_loaded_tokenizers_compatible,
    collect_tokenizer_runtime_evidence,
    require_local_model_weights,
    require_local_snapshot,
    verify_model_config_sha256,
    verify_tokenizer_runtime_evidence,
)
from experiments.aws_smoke.tokenization import (
    build_chat_token_pair,
    build_prompt_ids,
)


class FakeChatTokenizer:
    chat_template = "FAKE_PINNED_TEMPLATE"
    special_tokens_map = {
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "additional_special_tokens": ["<user>", "<assistant>"],
    }
    eos_token_id = 2

    def convert_tokens_to_ids(self, tokens):
        mapping = {
            "<eos>": 2,
            "<pad>": 0,
            "<user>": 10,
            "<assistant>": 11,
        }
        if isinstance(tokens, list):
            return [mapping[token] for token in tokens]
        return mapping[tokens]

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        assert tokenize is True
        user = conversation[0]["content"]
        template_token = 10 if self.chat_template == "FAKE_PINNED_TEMPLATE" else 12
        prompt = [template_token] + [30 + (ord(char) % 20) for char in user] + [11]
        if len(conversation) == 1:
            assert add_generation_prompt is True
            return prompt
        assert add_generation_prompt is False
        target = conversation[1]["content"]
        return prompt + [50 + (ord(char) % 20) for char in target] + [2]


def _snapshot(root: Path, revision: str) -> Path:
    path = root / "Qwen" / "Qwen2.5-0.5B-Instruct" / revision
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type":"qwen2"}\n', encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        '{"chat_template":"FAKE_PINNED_TEMPLATE"}\n',
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text(
        '{"version":"1.0"}\n',
        encoding="utf-8",
    )
    return path


def test_exact_local_revision_required(tmp_path: Path) -> None:
    revision = "a" * 40
    snapshot = _snapshot(tmp_path, revision)
    assert (
        require_local_snapshot(
            tmp_path,
            "Qwen/Qwen2.5-0.5B-Instruct",
            revision,
        )
        == snapshot
    )
    config_sha256 = hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest()
    assert verify_model_config_sha256(snapshot, config_sha256) == config_sha256
    with pytest.raises(ValueError, match="model config hash"):
        verify_model_config_sha256(snapshot, "f" * 64)
    with pytest.raises(FileNotFoundError, match="network fallback is forbidden"):
        require_local_snapshot(
            tmp_path,
            "Qwen/Qwen2.5-0.5B-Instruct",
            "b" * 40,
        )
    with pytest.raises(ValueError, match="40 lowercase"):
        require_local_snapshot(
            tmp_path,
            "Qwen/Qwen2.5-0.5B-Instruct",
            "main",
        )


def test_local_model_weights_required_before_loading(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "a" * 40)
    with pytest.raises(FileNotFoundError, match="local model weights"):
        require_local_model_weights(snapshot)
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"local-weights")
    assert require_local_model_weights(snapshot) == (weights,)
    weights.unlink()
    weights.symlink_to(snapshot / "missing-blob")
    with pytest.raises(FileNotFoundError, match="local model weights"):
        require_local_model_weights(snapshot)


def test_loaded_tokenizer_files_template_and_special_map_are_verified(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path, "a" * 40)
    tokenizer = FakeChatTokenizer()
    actual = collect_tokenizer_runtime_evidence(snapshot, tokenizer)
    verified = verify_tokenizer_runtime_evidence(
        snapshot_dir=snapshot,
        tokenizer=tokenizer,
        expected_tokenizer_sha256=actual.tokenizer_sha256,
        expected_chat_template_sha256=actual.chat_template_sha256,
        expected_special_token_map=actual.special_token_map,
    )
    assert verified.file_sha256.keys() == {
        "tokenizer_config.json",
        "tokenizer.json",
    }
    assert_loaded_tokenizers_compatible(actual, verified)
    with pytest.raises(ValueError, match="chat template"):
        verify_tokenizer_runtime_evidence(
            snapshot_dir=snapshot,
            tokenizer=tokenizer,
            expected_tokenizer_sha256=actual.tokenizer_sha256,
            expected_chat_template_sha256="f" * 64,
            expected_special_token_map=actual.special_token_map,
        )


def test_pinned_chat_template_builds_prefix_stable_pair_and_cap() -> None:
    tokenizer = FakeChatTokenizer()
    prompt_ids = build_prompt_ids(tokenizer, "abc")
    pair = build_chat_token_pair(
        tokenizer,
        "abc",
        "long-target",
        max_length=64,
        max_completion=5,
    )
    assert pair.input_ids[: len(prompt_ids)] == prompt_ids
    assert pair.labels[: len(prompt_ids)] == [-100] * len(prompt_ids)
    assert pair.completion_mask[: len(prompt_ids)] == [0.0] * len(prompt_ids)
    assert pair.completion_token_count == 5
    assert pair.original_completion_token_count > 5
    assert pair.completion_truncated is True
    assert pair.input_ids[-1] == tokenizer.eos_token_id


def test_template_attestation_is_not_unused(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "a" * 40)
    tokenizer = FakeChatTokenizer()
    evidence = collect_tokenizer_runtime_evidence(snapshot, tokenizer)
    # The attested template changes both the digest and actual token construction.
    original = build_prompt_ids(tokenizer, "x")
    tokenizer.chat_template = "DIFFERENT_TEMPLATE"
    changed = collect_tokenizer_runtime_evidence(snapshot, tokenizer)
    assert changed.chat_template_sha256 != evidence.chat_template_sha256
    assert build_prompt_ids(tokenizer, "x") != original


def test_snapshot_hash_changes_when_tokenizer_file_changes(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "a" * 40)
    tokenizer = FakeChatTokenizer()
    before = collect_tokenizer_runtime_evidence(snapshot, tokenizer)
    tokenizer_json = snapshot / "tokenizer.json"
    tokenizer_json.write_text(
        json.dumps({"version": "1.1"}) + "\n",
        encoding="utf-8",
    )
    after = collect_tokenizer_runtime_evidence(snapshot, tokenizer)
    assert after.tokenizer_sha256 != before.tokenizer_sha256
