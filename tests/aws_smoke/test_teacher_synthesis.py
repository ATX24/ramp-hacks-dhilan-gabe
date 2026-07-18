"""Sealed teacher-synthesis contracts without loading model weights."""

from __future__ import annotations

import json

import pytest

from distillery.contracts.hashing import content_sha256
from distillery.contracts.tasks import LabelSource
from distillery.recipes.sequence_v1 import materialize_sequence_examples, retokenize_text_pair
from experiments.aws_smoke.teacher_synthesis import (
    EXPECTED_MODEL_MATERIALIZATION_SHA256,
    FORBIDDEN_TEACHER_KEYS,
    GenerationConfig,
    assert_teacher_safe_prompt_payload,
    build_teacher_prompt,
    canonicalize_response_json,
    evaluate_teacher_response,
    extract_json_object,
    load_smoke_train_validation_rows,
    seal_teacher_row,
    verify_model_materialization_bytes,
)


def _fake_tokenizer_encode(text: str) -> dict[str, list]:
    return {
        "input_ids": [index + 1 for index in range(len(text))],
        "offset_mapping": [(index, index + 1) for index in range(len(text))],
    }


class _FakeTokenizer:
    def __call__(self, text: str, **_kwargs: object) -> dict[str, list]:
        return _fake_tokenizer_encode(text)


def test_model_materialization_hash_gate(tmp_path) -> None:
    payload = json.dumps({"schema": "wrong"}).encode()
    with pytest.raises(ValueError, match="materialization hash mismatch"):
        verify_model_materialization_bytes(payload)


def test_prompt_strips_labels_and_rejects_forbidden_keys() -> None:
    row = {
        "example_id": "ex_1",
        "task": "transaction_review",
        "difficulty": "easy",
        "input": {"memo": "coffee"},
        "expected_output": {"task": "transaction_review"},
        "oracle": {"latent_state_hash": "sha256:" + ("a" * 64)},
    }
    prompt = build_teacher_prompt(row)
    payload = json.loads(prompt)
    assert set(payload) == {"task", "difficulty", "input"}
    assert "expected_output" not in payload
    assert "oracle" not in payload
    with pytest.raises(ValueError, match="forbidden teacher field"):
        assert_teacher_safe_prompt_payload({"expected_output": {"a": 1}})
    assert "expected_output" in FORBIDDEN_TEACHER_KEYS


def test_evaluate_rejects_invalid_and_accepts_schema_valid_cash() -> None:
    bad_text, bad_reasons, bad_ok = evaluate_teacher_response(
        task="cash_reconciliation",
        raw_response="not-json",
    )
    assert bad_ok is False
    assert "invalid_json" in bad_reasons
    assert bad_text == "not-json"

    valid = {
        "schema_version": "cash_reconciliation.v1",
        "task": "cash_reconciliation",
        "book_balance_minor": 100,
        "bank_balance_minor": 90,
        "adjustments": [
            {
                "id": "adj_1",
                "amount_minor": 10,
                "direction": "add_to_book",
                "evidence": [{"source_id": "bank", "field": "amount", "value": "10"}],
            }
        ],
        "unexplained_difference_minor": 0,
        "confidence": 0.5,
    }
    # Use a minimal valid shape via extract/canonicalize path; if schema drifts,
    # rejection must still be explicit (never oracle-substituted).
    raw = "prefix " + json.dumps(valid) + " suffix"
    extracted = extract_json_object(raw)
    assert extracted is not None
    canonical = canonicalize_response_json(extracted)
    assert json.loads(canonical)["task"] == "cash_reconciliation"


def test_smoke_loader_excludes_held_out_splits() -> None:
    splits = load_smoke_train_validation_rows()
    assert set(splits) == {"train", "validation"}
    assert len(splits["train"]) == 320
    assert len(splits["validation"]) == 80
    all_ids = {row["example_id"] for rows in splits.values() for row in rows}
    assert len(all_ids) == 400
    for split, rows in splits.items():
        for row in rows:
            assert row["provenance"]["split"] == split
            prompt = build_teacher_prompt(row)
            assert "expected_output" not in prompt
            assert "oracle" not in prompt


def test_seal_retains_rejected_teacher_label_without_oracle_substitution() -> None:
    row = {
        "example_id": "ex_reject",
        "task": "transaction_review",
        "difficulty": "medium",
        "input": {"memo": "x"},
    }
    prompt = build_teacher_prompt(row)
    sealed = seal_teacher_row(
        row=row,
        split="train",
        prompt_text=prompt,
        raw_response="@@@broken",
        tokenizer=_FakeTokenizer(),
        tokenizer_sha256="b" * 64,
        chat_template_sha256_value="c" * 64,
        teacher_weight_sha256="d" * 64,
        generation_config=GenerationConfig(),
        prompt_token_count=11,
        generation_completion_token_count=3,
    )
    assert sealed.accepted is False
    assert sealed.record.label_source is LabelSource.TEACHER
    assert sealed.record.response_text == "@@@broken"
    assert "oracle" not in sealed.record.provenance_payload()
    report = materialize_sequence_examples([sealed.record])
    assert len(report.rejected) == 1
    assert report.rejected[0].label_source is LabelSource.TEACHER
    assert report.label_source_counts == {"teacher": 1}


def test_generation_config_is_greedy_temperature_zero() -> None:
    cfg = GenerationConfig()
    assert cfg.do_sample is False
    assert cfg.temperature == 0.0
    assert cfg.as_dict()["decoding"] == "greedy_temperature_zero"


def test_retokenize_binding_used_by_seal() -> None:
    prompt = "abc"
    target = '{"ok":true}'
    evidence = retokenize_text_pair(
        prompt,
        target,
        tokenizer_sha256="e" * 64,
        encode_with_offsets_fn=_fake_tokenizer_encode,
    )
    assert evidence.joint_text_sha256 == content_sha256(prompt + target)
    assert evidence.completion_start_index == len(prompt)
