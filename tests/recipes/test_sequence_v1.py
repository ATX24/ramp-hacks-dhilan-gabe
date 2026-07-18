"""sequence.v1 materialization and completion-only mask tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distillery.contracts.errors import DistilleryError, DistilleryErrorCode
from distillery.contracts.tasks import LabelSource
from distillery.recipes.base import (
    JointTokenizationEvidence,
    RecipeContext,
    ResponseRecord,
)
from distillery.recipes.sequence_v1 import (
    IGNORE_INDEX,
    SequenceV1Config,
    SequenceV1Recipe,
    build_completion_only_mask,
    build_completion_only_mask_from_joint,
    materialize_sequence_examples,
    retokenize_text_pair,
    validate_response_text,
)

REVISION = "a" * 40
DIGEST = "b" * 64


def _tokenization(
    prompt_text: str,
    selected_target_text: str,
) -> JointTokenizationEvidence:
    def encode_joint(text: str) -> dict[str, list]:
        return {
            "input_ids": [index + 1 for index in range(len(text))],
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }

    return retokenize_text_pair(
        prompt_text,
        selected_target_text,
        tokenizer_sha256=DIGEST,
        encode_with_offsets_fn=encode_joint,
    )


def _imported_record(
    *,
    example_id: str,
    response_text: str,
) -> ResponseRecord:
    prompt = "prompt"
    return ResponseRecord.seal(
        example_id=example_id,
        task="transaction_review",
        difficulty="medium",
        prompt_text=prompt,
        response_text=response_text,
        selected_target_text=response_text,
        label_source=LabelSource.IMPORTED,
        tokenization=_tokenization(prompt, response_text),
        imported_source_id=f"source-{example_id}",
        imported_source_sha256=DIGEST,
    )


def test_validate_response_accepts_json_object() -> None:
    assert validate_response_text('{"a": 1}') == ()


def test_validate_response_rejects_invalid_and_empty() -> None:
    assert "empty_response" in validate_response_text("   ")
    assert "invalid_json" in validate_response_text("not-json")
    assert "response_not_json_object" in validate_response_text("[1, 2]")


def test_materialize_separates_accepted_and_rejected() -> None:
    records = [
        _imported_record(
            example_id="ex_ok",
            response_text='{"task": "transaction_review"}',
        ),
        ResponseRecord.seal(
            example_id="ex_bad",
            task="variance_analysis",
            difficulty="hard",
            prompt_text="prompt",
            response_text="nope",
            selected_target_text="nope",
            label_source=LabelSource.TEACHER,
            tokenization=_tokenization("prompt", "nope"),
            teacher_model_id="teacher",
            teacher_revision=REVISION,
            generation_params={"temperature": 0.0},
            transformation_lineage=("teacher_generation",),
        ),
    ]
    report = materialize_sequence_examples(records)
    assert len(report.accepted) == 1
    assert report.accepted[0].example_id == "ex_ok"
    assert report.accepted[0].completion_token_count == len(
        '{"task": "transaction_review"}'
    )
    assert len(report.rejected) == 1
    assert report.label_source_counts["imported"] == 1
    assert report.label_source_counts["teacher"] == 1


def test_completion_only_mask_zeros_prompt_and_pad() -> None:
    mask = build_completion_only_mask(
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5],
        max_length=8,
        max_completion=4,
        pad_token_id=0,
    )
    assert mask.prompt_token_count == 3
    assert mask.completion_token_count == 2
    assert mask.input_ids == (1, 2, 3, 4, 5, 0, 0, 0)
    assert mask.attention_mask == (1, 1, 1, 1, 1, 0, 0, 0)
    assert mask.labels == (
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        4,
        5,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
    )
    assert mask.loss_mask == (0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert sum(mask.loss_mask) == 2.0


def test_completion_mask_rejects_total_and_completion_caps_separately() -> None:
    with pytest.raises(ValueError, match="max_length"):
        build_completion_only_mask(
            prompt_token_ids=[10, 11, 12, 13],
            completion_token_ids=[20, 21, 22],
            max_length=6,
            max_completion=3,
        )
    with pytest.raises(ValueError, match="max_completion"):
        build_completion_only_mask(
            prompt_token_ids=[10],
            completion_token_ids=[20, 21, 22],
            max_length=8,
            max_completion=2,
        )


def test_retokenize_jointly_handles_boundary_sensitive_token() -> None:
    calls: list[str] = []

    def boundary_sensitive_encode(text: str) -> dict[str, list]:
        calls.append(text)
        assert text == "abc"
        return {
            "input_ids": [99],
            "offset_mapping": [(0, 3)],
        }

    evidence = retokenize_text_pair(
        "ab",
        "c",
        tokenizer_sha256=DIGEST,
        encode_with_offsets_fn=boundary_sensitive_encode,
    )
    assert calls == ["abc"]
    assert evidence.completion_start_index == 0
    assert evidence.prompt_token_count == 0
    assert evidence.completion_token_count == 1
    mask = build_completion_only_mask_from_joint(
        evidence,
        max_length=4,
        max_completion=1,
    )
    assert mask.labels[0] == 99


def test_sequence_recipe_requires_pinned_revision() -> None:
    recipe = SequenceV1Recipe(SequenceV1Config())
    ctx = RecipeContext(
        run_id="run_test",
        seed=17,
        max_length=512,
        max_completion=160,
        student_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        student_revision="main",
    )
    with pytest.raises(DistilleryError) as exc:
        recipe.validate_capabilities(ctx)
    assert exc.value.code is DistilleryErrorCode.MODEL_REVISION_UNPINNED


def test_sequence_recipe_materialize_fails_when_all_rejected() -> None:
    recipe = SequenceV1Recipe()
    ctx = RecipeContext(
        run_id="run_test",
        seed=17,
        max_length=512,
        max_completion=160,
        student_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        student_revision=REVISION,
    )
    records = [
        ResponseRecord.seal(
            example_id="ex_x",
            task="cash_reconciliation",
            difficulty="easy",
            prompt_text="p",
            response_text="not-json",
            selected_target_text="not-json",
            label_source=LabelSource.ORACLE,
            tokenization=_tokenization("p", "not-json"),
            oracle_generator_revision=REVISION,
            oracle_latent_state_hash="sha256:" + DIGEST,
            rule_ids=("oracle-rule-v1",),
        )
    ]
    with pytest.raises(DistilleryError) as exc:
        recipe.materialize(records, context=ctx)
    assert exc.value.code is DistilleryErrorCode.UNSUPPORTED_LABEL_SOURCE


def test_teacher_provenance_requires_pinned_model_identity() -> None:
    with pytest.raises(ValidationError, match="teacher provenance requires"):
        ResponseRecord(
            example_id="ex_teacher",
            task="transaction_review",
            difficulty="easy",
            prompt_text="p",
            response_text='{"ok":true}',
            selected_target_text='{"ok":true}',
            label_source=LabelSource.TEACHER,
            tokenization=_tokenization("p", '{"ok":true}'),
            record_sha256="0" * 64,
        )
    with pytest.raises(ValidationError, match="40 lowercase hex"):
        ResponseRecord(
            example_id="ex_teacher",
            task="transaction_review",
            difficulty="easy",
            prompt_text="p",
            response_text='{"ok":true}',
            selected_target_text='{"ok":true}',
            label_source=LabelSource.TEACHER,
            tokenization=_tokenization("p", '{"ok":true}'),
            teacher_model_id="teacher",
            teacher_revision="main",
            generation_params={"temperature": 0.0},
            transformation_lineage=("teacher_generation",),
            record_sha256="0" * 64,
        )


def test_imported_and_oracle_provenance_cannot_lie() -> None:
    with pytest.raises(ValidationError, match="imported provenance requires"):
        ResponseRecord(
            example_id="ex_imported",
            task="transaction_review",
            difficulty="easy",
            prompt_text="p",
            response_text='{"ok":true}',
            selected_target_text='{"ok":true}',
            label_source=LabelSource.IMPORTED,
            tokenization=_tokenization("p", '{"ok":true}'),
            record_sha256="0" * 64,
        )
    with pytest.raises(ValidationError, match="another source"):
        ResponseRecord(
            example_id="ex_oracle",
            task="variance_analysis",
            difficulty="hard",
            prompt_text="p",
            response_text='{"ok":true}',
            selected_target_text='{"ok":true}',
            label_source=LabelSource.ORACLE,
            tokenization=_tokenization("p", '{"ok":true}'),
            oracle_generator_revision=REVISION,
            oracle_latent_state_hash="sha256:" + DIGEST,
            rule_ids=("oracle-rule-v1",),
            teacher_model_id="teacher",
            teacher_revision=REVISION,
            record_sha256="0" * 64,
        )


def test_record_hash_binds_target_tokenization_and_provenance() -> None:
    record = _imported_record(
        example_id="ex_bound",
        response_text='{"ok":true}',
    )
    payload = record.model_dump(mode="json")
    payload["selected_target_text"] = '{"ok":false}'
    with pytest.raises(
        ValidationError,
        match="record_sha256|joint tokenization|transformation_lineage",
    ):
        ResponseRecord.model_validate(payload)

    payload = record.model_dump(mode="json")
    payload["imported_source_id"] = "different-source"
    with pytest.raises(ValidationError, match="record_sha256"):
        ResponseRecord.model_validate(payload)

    record.generation_params["mutated_after_seal"] = True
    with pytest.raises(ValueError, match="record_sha256"):
        materialize_sequence_examples([record])


def test_materialization_preserves_complete_source_provenance() -> None:
    target = '{"ok":true}'
    imported = _imported_record(
        example_id="ex_imported",
        response_text=target,
    )
    teacher = ResponseRecord.seal(
        example_id="ex_teacher",
        task="transaction_review",
        difficulty="easy",
        prompt_text="p",
        response_text=target,
        selected_target_text=target,
        label_source=LabelSource.TEACHER,
        tokenization=_tokenization("p", target),
        teacher_model_id="teacher",
        teacher_revision=REVISION,
        generation_params={"temperature": 0.0},
        transformation_lineage=("teacher_generation",),
    )
    oracle = ResponseRecord.seal(
        example_id="ex_oracle",
        task="variance_analysis",
        difficulty="hard",
        prompt_text="p",
        response_text=target,
        selected_target_text=target,
        label_source=LabelSource.ORACLE,
        tokenization=_tokenization("p", target),
        oracle_generator_revision=REVISION,
        oracle_latent_state_hash="sha256:" + DIGEST,
        rule_ids=("oracle-rule-v1",),
    )
    report = materialize_sequence_examples([imported, teacher, oracle])
    by_id = {example.example_id: example for example in report.accepted}
    for source in (imported, teacher, oracle):
        materialized = by_id[source.example_id]
        assert materialized.record_sha256 == source.record_sha256
        assert materialized.source_response_text == source.response_text
        assert materialized.selected_target_text == source.selected_target_text
        assert materialized.provenance == source.provenance_payload()


def test_teacher_and_oracle_require_complete_generation_provenance() -> None:
    target = '{"ok":true}'
    common = {
        "task": "transaction_review",
        "difficulty": "easy",
        "prompt_text": "p",
        "response_text": target,
        "selected_target_text": target,
        "tokenization": _tokenization("p", target),
        "record_sha256": "0" * 64,
    }
    with pytest.raises(ValidationError, match="generation_params"):
        ResponseRecord(
            example_id="ex_teacher",
            label_source=LabelSource.TEACHER,
            teacher_model_id="teacher",
            teacher_revision=REVISION,
            transformation_lineage=("teacher_generation",),
            **common,
        )
    with pytest.raises(ValidationError, match="rule_ids"):
        ResponseRecord(
            example_id="ex_oracle",
            label_source=LabelSource.ORACLE,
            oracle_generator_revision=REVISION,
            oracle_latent_state_hash="sha256:" + DIGEST,
            **common,
        )


def test_materialization_rejects_999_token_completion_under_160_cap() -> None:
    target = '{"value":"' + ("x" * 987) + '"}'
    record = _imported_record(example_id="ex_999", response_text=target)
    assert record.completion_token_count == 999
    report = materialize_sequence_examples(
        [record],
        config=SequenceV1Config(max_completion=160, max_length=2_000),
    )
    assert report.accepted == ()
    assert "completion_token_count_exceeds_max" in (
        report.rejected[0].rejection_reasons
    )


def test_completion_mask_rejects_malformed_token_ids() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        build_completion_only_mask(
            prompt_token_ids=[1, -1],
            completion_token_ids=[2],
            max_length=4,
            max_completion=2,
        )
