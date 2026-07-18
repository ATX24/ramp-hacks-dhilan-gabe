"""Adversarial tests for validate-only boundaries and execute hard gates."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from distillery.contracts.errors import DistilleryError, DistilleryErrorCode, ErrorPayload
from distillery.contracts.manifest import (
    ManifestCost,
    ManifestDatasetRef,
    ManifestModels,
    ManifestModelSpec,
    ManifestOutput,
    ManifestProofProtocol,
    ManifestRecipe,
    ManifestRuntime,
    ManifestTraining,
    SealedRunManifest,
)
from distillery.contracts.recipes import AutoResolverInput
from distillery.contracts.tasks import LabelSource
from distillery.recipes.auto import resolve_recipe
from distillery.recipes.base import ResponseRecord
from distillery.recipes.logit_v1 import memory_dry_run_evidence_sha256
from distillery.recipes.sequence_v1 import retokenize_text_pair
from distillery.training.batching import SamplerExample, plan_batches
from distillery.training.entrypoint import (
    CAPABILITY_EVIDENCE_KEY,
    EXECUTE_ACKNOWLEDGEMENT,
    build_response_file_evidence,
    canonical_response_jsonl,
    capability_binding_sha256,
    length_configuration_sha256,
    main,
    model_configuration_sha256,
    run_entrypoint,
    training_configuration_sha256,
)

TEACHER_REVISION = "1" * 40
STUDENT_REVISION = "2" * 40
TOKENIZER_DIGEST = "a" * 64
CHAT_DIGEST = "b" * 64
SOURCE_DIGEST = "c" * 64
SPECIAL_TOKEN_MAP = {"pad_token_id": 0, "eos_token_id": 1}


def _digest(n: int = 0) -> str:
    return f"{n:064x}"


def _tokenization(
    prompt: str,
    target: str,
    *,
    completion_count: int,
    prompt_count: int = 1,
    tokenizer_sha256: str = TOKENIZER_DIGEST,
):
    def encode_joint(text: str) -> dict[str, list]:
        boundary = len(prompt)
        assert text == prompt + target
        return {
            "input_ids": list(
                range(1, prompt_count + completion_count + 1)
            ),
            "offset_mapping": [
                *[(0, boundary) for _ in range(prompt_count)],
                *[(boundary, len(text)) for _ in range(completion_count)],
            ],
        }

    return retokenize_text_pair(
        prompt,
        target,
        tokenizer_sha256=tokenizer_sha256,
        encode_with_offsets_fn=encode_joint,
    )


def _imported_record(
    *,
    index: int,
    task: str,
    difficulty: str,
    completion_count: int,
    tokenizer_sha256: str = TOKENIZER_DIGEST,
) -> ResponseRecord:
    prompt = f"prompt-{index}"
    response = json.dumps({"task": task, "index": index})
    return ResponseRecord.seal(
        example_id=f"ex_{index:02d}",
        task=task,
        difficulty=difficulty,
        prompt_text=prompt,
        response_text=response,
        selected_target_text=response,
        label_source=LabelSource.IMPORTED,
        tokenization=_tokenization(
            prompt,
            response,
            completion_count=completion_count,
            tokenizer_sha256=tokenizer_sha256,
        ),
        imported_source_id=f"fixture-{index}",
        imported_source_sha256=SOURCE_DIGEST,
    )


def _records() -> list[ResponseRecord]:
    """20 records exactly matching rounded 45/45/10 and 30/40/30 strata."""
    task_difficulties = {
        "transaction_review": ["easy"] * 3 + ["medium"] * 3 + ["hard"] * 3,
        "variance_analysis": ["easy"] * 3 + ["medium"] * 3 + ["hard"] * 3,
        "cash_reconciliation": ["easy", "medium"],
    }
    records: list[ResponseRecord] = []
    index = 0
    for task, difficulties in task_difficulties.items():
        for difficulty in difficulties:
            records.append(
                _imported_record(
                    index=index,
                    task=task,
                    difficulty=difficulty,
                    completion_count=10 + index,
                )
            )
            index += 1
    return records


def _sampler_hash(records: list[ResponseRecord]) -> str:
    examples = [
        SamplerExample(
            example_id=record.example_id,
            task=record.task,
            difficulty=record.difficulty,
            completion_tokens=record.completion_token_count,
            prompt_tokens=record.prompt_token_count,
            total_tokens=record.total_token_count,
            completion_token_source=record.completion_token_count_source,
            completion_tokenizer_sha256=record.completion_tokenizer_sha256,
            record_sha256=record.record_sha256,
        )
        for record in records
    ]
    return plan_batches(examples, seed=17).sampler_order_hash


def _write_records(path: Path, records: list[ResponseRecord] | None = None) -> Path:
    selected = records if records is not None else _records()
    path.write_bytes(canonical_response_jsonl(selected))
    return path


def _base_manifest(
    *,
    requested: str = "sequence.v1",
    resolved: str = "sequence.v1",
    resolver_reasons: tuple[str, ...] = ("explicit_request",),
    sampler_hash: str | None = None,
    records: list[ResponseRecord] | None = None,
    auto_input: AutoResolverInput | None = None,
) -> SealedRunManifest:
    selected_records = records if records is not None else _records()
    response_evidence = build_response_file_evidence(
        selected_records,
        max_completion=160,
        max_length=512,
    )
    qlora_payload: dict = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "logit_temperature": 2.0,
        "kd_weight": 0.7,
        "hard_ce_weight": 0.3,
        "vocab_chunk": 4096,
        "max_completion": 160,
    }
    manifest = SealedRunManifest(
        run_id="run_entrypoint1",
        created_at=datetime(2026, 7, 18, tzinfo=UTC),
        dataset=ManifestDatasetRef(
            dataset_id="ds_finance1",
            uri="s3://bucket/datasets/ds_finance1",
            sha256=_digest(3),
            split_sha256={"train": _digest(4), "validation": _digest(5)},
        ),
        models=ManifestModels(
            teacher=ManifestModelSpec(
                id="Qwen/Qwen2.5-1.5B-Instruct",
                revision=TEACHER_REVISION,
                tokenizer_sha256=TOKENIZER_DIGEST,
                chat_template_sha256=CHAT_DIGEST,
            ),
            student=ManifestModelSpec(
                id="Qwen/Qwen2.5-0.5B-Instruct",
                revision=STUDENT_REVISION,
                tokenizer_sha256=TOKENIZER_DIGEST,
                chat_template_sha256=CHAT_DIGEST,
            ),
        ),
        recipe=ManifestRecipe(
            requested="sequence.v1",
            resolved="sequence.v1",
            resolver_reasons=("explicit_request",),
        ),
        training=ManifestTraining(
            seed=17,
            max_steps=30,
            token_budget=0,
            max_length=512,
            completion_evidence=response_evidence.model_dump(mode="json"),
            qlora=qlora_payload,
        ),
        proof_protocol=ManifestProofProtocol(
            id="finance-proof.v1",
            sha256=_digest(6),
        ),
        runtime=ManifestRuntime(
            backend="local",
            region="us-east-1",
            instance_type="ml.g5.xlarge",
            image_digest="sha256:" + _digest(7),
        ),
        cost=ManifestCost(
            max_run_usd=25.0,
            estimate_low_usd=1.0,
            estimate_high_usd=5.0,
        ),
        output=ManifestOutput(prefix="s3://bucket/runs/run_entrypoint1"),
        package_lock_hash=_digest(8),
        source_revision="deadbeef",
        sampler_order_hash=sampler_hash or _sampler_hash(selected_records),
    )
    if requested == "sequence.v1" and resolved == "sequence.v1":
        return manifest

    target_recipe = ManifestRecipe(
        requested=requested,
        resolved=resolved,
        resolver_reasons=resolver_reasons,
    )
    target = BaseModel.model_copy(
        manifest,
        update={"recipe": target_recipe},
    )
    if resolved == "do_not_distill":
        training_without_completion = BaseModel.model_copy(
            target.training,
            update={"completion_evidence": None},
        )
        target = BaseModel.model_copy(
            target,
            update={"training": training_without_completion},
        )
    capability_input = auto_input
    if requested == "auto" and capability_input is None:
        capability_input = AutoResolverInput(usable_responses_exist=True)
    elif resolved == "logit.v1" and capability_input is None:
        capability_input = AutoResolverInput(
            local_white_box=True,
            tokenizer_fingerprint_match=True,
            special_token_map_match=True,
            chat_template_compatible=True,
            memory_dry_run_ok=True,
        )
    target = _with_capability_evidence(
        target,
        auto_input=capability_input,
        include_special_tokens=resolved == "logit.v1",
        include_memory=resolved == "logit.v1",
    )
    return SealedRunManifest.model_validate(
        target.model_dump(mode="json", warnings=False)
    )


def _with_capability_evidence(
    manifest: SealedRunManifest,
    *,
    auto_input: AutoResolverInput | None = None,
    include_special_tokens: bool = True,
    include_memory: bool = True,
    memory_overrides: dict | None = None,
) -> SealedRunManifest:
    evidence: dict = {
        "schema_version": "distillery.training_capabilities.v1",
    }
    if auto_input is not None:
        evidence["auto_resolver_input"] = auto_input.model_dump(mode="json")
    if include_special_tokens:
        evidence["special_token_maps"] = {
            "teacher": SPECIAL_TOKEN_MAP,
            "student": SPECIAL_TOKEN_MAP,
        }
    if include_memory:
        memory = {
            "schema_version": "distillery.memory_dry_run.v2",
            "passed": True,
            "binding_sha256": capability_binding_sha256(
                manifest,
                teacher_special_token_map=(
                    SPECIAL_TOKEN_MAP if include_special_tokens else {}
                ),
                student_special_token_map=(
                    SPECIAL_TOKEN_MAP if include_special_tokens else {}
                ),
                auto_resolver_input=(
                    auto_input.model_dump(mode="json")
                    if auto_input is not None
                    else None
                ),
            ),
            "training_config_sha256": training_configuration_sha256(manifest),
            "teacher_model_config_sha256": model_configuration_sha256(
                manifest.models.teacher
            ),
            "student_model_config_sha256": model_configuration_sha256(
                manifest.models.student
            ),
            "length_config_sha256": length_configuration_sha256(manifest),
            "runtime_image_digest": manifest.runtime.image_digest,
            "instance_type": manifest.runtime.instance_type,
            "recipe_id": "logit.v1",
            "teacher_model_id": manifest.models.teacher.id,
            "teacher_revision": manifest.models.teacher.revision,
            "student_model_id": manifest.models.student.id,
            "student_revision": manifest.models.student.revision,
            "max_length": manifest.training.max_length,
            "max_completion": 160,
            "vocab_chunk_size": 4096,
            "peak_memory_bytes": 20_000_000_000,
            "capacity_memory_bytes": 24_000_000_000,
            "headroom_bytes": 4_000_000_000,
            "device_type": "precomputed-a10g-profile",
            "probe_id": "probe-synthetic-1",
        }
        memory.update(memory_overrides or {})
        memory["evidence_sha256"] = memory_dry_run_evidence_sha256(memory)
        evidence["memory_dry_run"] = memory
    qlora = dict(manifest.training.qlora)
    qlora[CAPABILITY_EVIDENCE_KEY] = evidence
    training = BaseModel.model_copy(
        manifest.training,
        update={"qlora": qlora},
    )
    return BaseModel.model_copy(manifest, update={"training": training})


def _manifest(
    *,
    requested: str = "sequence.v1",
    resolved: str = "sequence.v1",
    auto_input: AutoResolverInput | None = None,
    include_logit_evidence: bool = True,
    sampler_hash: str | None = None,
    records: list[ResponseRecord] | None = None,
) -> SealedRunManifest:
    reasons = ("explicit_request",)
    if requested == "auto" and auto_input is not None:
        reasons = resolve_recipe("auto", auto_input=auto_input).reasons
    manifest = _base_manifest(
        requested=requested,
        resolved=resolved,
        resolver_reasons=reasons,
        sampler_hash=sampler_hash,
        records=records,
        auto_input=auto_input,
    )
    if resolved == "logit.v1" and not include_logit_evidence:
        qlora = dict(manifest.training.qlora)
        qlora.pop(CAPABILITY_EVIDENCE_KEY, None)
        training = BaseModel.model_copy(
            manifest.training,
            update={"qlora": qlora},
        )
        manifest = BaseModel.model_copy(
            manifest,
            update={"training": training},
        )
    return manifest


def test_validate_only_default_does_not_execute(tmp_path: Path) -> None:
    responses = _write_records(tmp_path / "responses.jsonl")
    result = run_entrypoint(
        manifest=_manifest(),
        responses_path=responses,
        output_dir=tmp_path / "out",
    )
    assert result.executed is False
    assert result.mode == "validate_only"
    payload = json.loads(
        (tmp_path / "out" / "validation_result.json").read_text(encoding="utf-8")
    )
    assert payload["mode"] == "validation_only"
    assert payload["executed"] is False
    assert "execute" not in payload
    materialization = json.loads(
        (tmp_path / "out" / "materialization.json").read_text(encoding="utf-8")
    )
    assert materialization["executed"] is False
    assert materialization["completion_token_count_source"] == "student_tokenizer"


def test_execute_requires_exact_acknowledgement_and_writes_nothing(
    tmp_path: Path,
) -> None:
    responses = _write_records(tmp_path / "responses.jsonl")
    output = tmp_path / "out"
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(
            manifest=_manifest(),
            responses_path=responses,
            execute=True,
            output_dir=output,
        )
    assert exc.value.code is DistilleryErrorCode.RECIPE_INCOMPATIBLE
    assert not output.exists()

    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(
            manifest=_manifest(),
            responses_path=responses,
            execute=True,
            execute_acknowledgement=EXECUTE_ACKNOWLEDGEMENT,
            output_dir=output,
        )
    assert exc.value.code is DistilleryErrorCode.CAPABILITY_UNAVAILABLE
    assert exc.value.payload.details["sidecars_written"] is False
    assert not output.exists()


def test_cli_defaults_to_validate_only_and_ignores_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    responses = _write_records(tmp_path / "responses.jsonl")
    manifest_path.write_text(_manifest().model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("DISTILLERY_EXECUTE", "1")
    monkeypatch.setenv("EXECUTE", "true")
    buf = io.StringIO()
    code = main(
        [
            "--manifest",
            str(manifest_path),
            "--responses",
            str(responses),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        stdout=buf,
    )
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["executed"] is False
    assert payload["mode"] == "validate_only"


def test_cli_execute_hard_stops_before_sidecars(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    responses = _write_records(tmp_path / "responses.jsonl")
    manifest_path.write_text(_manifest().model_dump_json(), encoding="utf-8")
    output = tmp_path / "out"
    buf = io.StringIO()
    code = main(
        [
            "--manifest",
            str(manifest_path),
            "--responses",
            str(responses),
            "--output-dir",
            str(output),
            "--execute",
            "--execute-acknowledgement",
            EXECUTE_ACKNOWLEDGEMENT,
        ],
        stdout=buf,
    )
    assert code == 1
    payload = json.loads(buf.getvalue())
    assert payload["code"] == DistilleryErrorCode.CAPABILITY_UNAVAILABLE.value
    assert not output.exists()


def test_logit_manifest_requires_special_tokens_and_memory(
    tmp_path: Path,
) -> None:
    responses = _write_records(tmp_path / "responses.jsonl")
    valid = _manifest(requested="logit.v1", resolved="logit.v1")
    result = run_entrypoint(manifest=valid, responses_path=responses)
    assert result.recipe == "logit.v1"
    assert result.load_plan["teacher"] is not None

    missing_special = _with_capability_evidence(
        _base_manifest(requested="logit.v1", resolved="logit.v1"),
        include_special_tokens=False,
        include_memory=True,
    )
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(manifest=missing_special, responses_path=responses)
    assert exc.value.code is DistilleryErrorCode.TOKENIZER_MISMATCH

    missing_memory = _with_capability_evidence(
        _base_manifest(requested="logit.v1", resolved="logit.v1"),
        include_special_tokens=True,
        include_memory=False,
    )
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(manifest=missing_memory, responses_path=responses)
    assert exc.value.code is DistilleryErrorCode.MEMORY_DRY_RUN_FAILED


def test_logit_memory_evidence_is_bound_to_exact_manifest(tmp_path: Path) -> None:
    responses = _write_records(tmp_path / "responses.jsonl")
    mismatched = _with_capability_evidence(
        _base_manifest(requested="logit.v1", resolved="logit.v1"),
        memory_overrides={"binding_sha256": "f" * 64},
    )
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(manifest=mismatched, responses_path=responses)
    assert exc.value.code is DistilleryErrorCode.MEMORY_DRY_RUN_FAILED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device_type", "different-device"),
        ("instance_type", "different-instance"),
        ("peak_memory_bytes", 19_000_000_000),
        ("probe_id", "different-probe"),
        ("teacher_model_config_sha256", "d" * 64),
        ("runtime_image_digest", "sha256:" + ("e" * 64)),
        ("training_config_sha256", "f" * 64),
        ("length_config_sha256", "1" * 64),
    ],
)
def test_memory_evidence_rejects_post_binding_mutation(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    responses = _write_records(tmp_path / "responses.jsonl")
    manifest = _manifest(requested="logit.v1", resolved="logit.v1")
    qlora = dict(manifest.training.qlora)
    capability = dict(qlora[CAPABILITY_EVIDENCE_KEY])
    memory = dict(capability["memory_dry_run"])
    memory[field] = value
    capability["memory_dry_run"] = memory
    qlora[CAPABILITY_EVIDENCE_KEY] = capability
    training = BaseModel.model_copy(
        manifest.training,
        update={"qlora": qlora},
    )
    mutated = BaseModel.model_copy(
        manifest,
        update={"training": training},
    )

    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(manifest=mutated, responses_path=responses)
    assert exc.value.code is DistilleryErrorCode.MEMORY_DRY_RUN_FAILED
    assert "validation_errors" in exc.value.payload.details


def test_auto_resolution_requires_and_recomputes_embedded_input(
    tmp_path: Path,
) -> None:
    responses = _write_records(tmp_path / "responses.jsonl")
    ambiguous = _base_manifest(
        requested="auto",
        resolved="sequence.v1",
        resolver_reasons=("usable_responses_present", "no_teacher_calls_required"),
    )
    qlora = dict(ambiguous.training.qlora)
    qlora.pop(CAPABILITY_EVIDENCE_KEY, None)
    training = BaseModel.model_copy(
        ambiguous.training,
        update={"qlora": qlora},
    )
    ambiguous = BaseModel.model_copy(
        ambiguous,
        update={"training": training},
    )
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(manifest=ambiguous, responses_path=responses)
    assert exc.value.code is DistilleryErrorCode.AUTO_RESOLVER_FAILED
    assert "integration_requirement" in exc.value.payload.details

    auto_input = AutoResolverInput(usable_responses_exist=True)
    valid = _manifest(
        requested="auto",
        resolved="sequence.v1",
        auto_input=auto_input,
    )
    assert run_entrypoint(manifest=valid, responses_path=responses).recipe == "sequence.v1"

    truthful = _manifest(
        requested="auto",
        resolved="sequence.v1",
        auto_input=auto_input,
    )
    lying_recipe = BaseModel.model_copy(
        truthful.recipe,
        update={"resolved": "logit.v1"},
    )
    lying = BaseModel.model_copy(
        truthful,
        update={"recipe": lying_recipe},
    )
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(manifest=lying, responses_path=responses)
    assert exc.value.code is DistilleryErrorCode.RECIPE_INCOMPATIBLE


def test_auto_logit_claim_requires_matching_full_capability_evidence(
    tmp_path: Path,
) -> None:
    responses = _write_records(tmp_path / "responses.jsonl")
    auto_input = AutoResolverInput(
        local_white_box=True,
        tokenizer_fingerprint_match=True,
        special_token_map_match=True,
        chat_template_compatible=True,
        memory_dry_run_ok=True,
    )
    manifest = _manifest(
        requested="auto",
        resolved="logit.v1",
        auto_input=auto_input,
    )
    result = run_entrypoint(manifest=manifest, responses_path=responses)
    assert result.recipe == "logit.v1"


def test_do_not_distill_remains_non_trainable(tmp_path: Path) -> None:
    responses = _write_records(tmp_path / "responses.jsonl")
    auto_input = AutoResolverInput(cheaper_baseline_satisfies_gate=True)
    resolution = resolve_recipe("auto", auto_input=auto_input)
    manifest = _base_manifest(
        requested="auto",
        resolved="do_not_distill",
        resolver_reasons=resolution.reasons,
        auto_input=auto_input,
    )
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(manifest=manifest, responses_path=responses)
    assert exc.value.code is DistilleryErrorCode.CAPABILITY_UNAVAILABLE


def test_sealed_sampler_hash_lie_fails_without_normalization(tmp_path: Path) -> None:
    responses = _write_records(tmp_path / "responses.jsonl")
    manifest = _manifest(sampler_hash="f" * 64)
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(
            manifest=manifest,
            responses_path=responses,
            output_dir=tmp_path / "out",
        )
    assert exc.value.code is DistilleryErrorCode.INVALID_DATASET
    assert exc.value.payload.details["sealed_value_preserved"] is True
    assert exc.value.payload.details["sealed_sampler_order_hash"] == "f" * 64
    assert not (tmp_path / "out").exists()


def test_missing_joint_tokenization_evidence_fails_loud(
    tmp_path: Path,
) -> None:
    records = _records()
    payloads = [record.model_dump(mode="json") for record in records]
    payloads[0].pop("tokenization")
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        "\n".join(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
            for payload in payloads
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(manifest=_manifest(), responses_path=responses)
    assert exc.value.code is DistilleryErrorCode.INVALID_DATASET
    assert exc.value.payload.details["line"] == 1


def test_completion_counts_are_bound_to_student_tokenizer(tmp_path: Path) -> None:
    records = [
        ResponseRecord.seal(
            **{
                **record.canonical_payload(),
                "tokenization": _tokenization(
                    record.prompt_text,
                    record.selected_target_text,
                    completion_count=record.completion_token_count,
                    tokenizer_sha256="f" * 64,
                ).model_dump(mode="json"),
            }
        )
        for record in _records()
    ]
    responses = _write_records(tmp_path / "responses.jsonl", records)
    manifest = _manifest()
    wrong_evidence = build_response_file_evidence(
        records,
        max_completion=160,
        max_length=512,
    )
    training = BaseModel.model_copy(
        manifest.training,
        update={"completion_evidence": wrong_evidence.model_dump(mode="json")},
    )
    manifest = BaseModel.model_copy(
        manifest,
        update={"training": training},
    )
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(
            manifest=manifest,
            responses_path=responses,
        )
    assert exc.value.code is DistilleryErrorCode.INVALID_DATASET
    assert exc.value.payload.details["expected_student_tokenizer_sha256"] == (
        TOKENIZER_DIGEST
    )


def test_max_completion_rejects_999_token_record_before_batching(
    tmp_path: Path,
) -> None:
    records = _records()
    records[0] = _imported_record(
        index=0,
        task="transaction_review",
        difficulty="easy",
        completion_count=999,
    )
    with pytest.raises(ValueError, match="max_completion"):
        build_response_file_evidence(
            records,
            max_completion=160,
            max_length=2_000,
        )
    evidence = build_response_file_evidence(
        records,
        max_completion=999,
        max_length=2_000,
    )
    manifest = _manifest()
    training = BaseModel.model_copy(
        manifest.training,
        update={"completion_evidence": evidence.model_dump(mode="json")},
    )
    manifest = BaseModel.model_copy(
        manifest,
        update={"training": training},
    )
    responses = _write_records(tmp_path / "responses.jsonl", records)
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(
            manifest=manifest,
            responses_path=responses,
        )
    assert exc.value.code is DistilleryErrorCode.INVALID_DATASET
    assert exc.value.payload.details["violations"][0] == {
        "example_id": "ex_00",
        "completion_token_count": 999,
        "max_completion": 160,
    }


def test_total_length_cap_is_independent_from_completion_cap(
    tmp_path: Path,
) -> None:
    records = _records()
    base = records[0]
    payload = base.canonical_payload()
    payload["tokenization"] = _tokenization(
        base.prompt_text,
        base.selected_target_text,
        completion_count=base.completion_token_count,
        prompt_count=503,
    ).model_dump(mode="json")
    records[0] = ResponseRecord.seal(**payload)
    assert records[0].completion_token_count == 10
    assert records[0].total_token_count == 513
    with pytest.raises(ValueError, match="max_length"):
        build_response_file_evidence(
            records,
            max_completion=160,
            max_length=512,
        )
    evidence = build_response_file_evidence(
        records,
        max_completion=160,
        max_length=513,
    )
    manifest = _manifest()
    training = BaseModel.model_copy(
        manifest.training,
        update={"completion_evidence": evidence.model_dump(mode="json")},
    )
    manifest = BaseModel.model_copy(
        manifest,
        update={"training": training},
    )
    responses = _write_records(tmp_path / "responses.jsonl", records)
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(
            manifest=manifest,
            responses_path=responses,
        )
    assert exc.value.code is DistilleryErrorCode.INVALID_DATASET
    assert exc.value.payload.details["violations"][0]["max_length"] == 512


@pytest.mark.parametrize("mutation", ["target", "provenance"])
def test_prior_file_seal_rejects_content_or_provenance_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    original = _records()
    changed = list(original)
    payload = original[0].canonical_payload()
    if mutation == "target":
        target = '{"task":"changed","i":0}'
        payload["selected_target_text"] = target
        payload["transformation_lineage"] = ["target_selection"]
        payload["tokenization"] = _tokenization(
            original[0].prompt_text,
            target,
            completion_count=original[0].completion_token_count,
        ).model_dump(mode="json")
    else:
        payload["imported_source_id"] = "different-source"
    changed[0] = ResponseRecord.seal(**payload)
    assert changed[0].example_id == original[0].example_id
    assert (
        changed[0].completion_token_count
        == original[0].completion_token_count
    )
    responses = _write_records(tmp_path / "responses.jsonl", changed)

    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(manifest=_manifest(), responses_path=responses)
    assert exc.value.code is DistilleryErrorCode.INVALID_DATASET
    assert exc.value.payload.details["content_and_provenance_bound"] is True
    assert "record_sha256" in exc.value.payload.details["violations"]


def test_completion_seal_cannot_claim_rejected_target_is_accepted(
    tmp_path: Path,
) -> None:
    records = _records()
    payload = records[0].canonical_payload()
    payload["response_text"] = "not-json"
    payload["selected_target_text"] = "not-json"
    payload["tokenization"] = _tokenization(
        records[0].prompt_text,
        "not-json",
        completion_count=records[0].completion_token_count,
    ).model_dump(mode="json")
    records[0] = ResponseRecord.seal(**payload)
    responses = _write_records(tmp_path / "responses.jsonl", records)

    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(
            manifest=_manifest(records=records),
            responses_path=responses,
        )
    assert exc.value.code is DistilleryErrorCode.INVALID_DATASET
    assert exc.value.payload.details["rejected_example_ids"] == ("ex_00",)


def test_trainer_defensively_rejects_non_sha_revision(tmp_path: Path) -> None:
    responses = _write_records(tmp_path / "responses.jsonl")
    manifest = _manifest()
    bad_student = BaseModel.model_copy(
        manifest.models.student,
        update={"revision": "student-main"},
    )
    bad_models = BaseModel.model_copy(
        manifest.models,
        update={"student": bad_student},
    )
    bad_manifest = BaseModel.model_copy(
        manifest,
        update={"models": bad_models},
    )
    with pytest.raises(DistilleryError) as exc:
        run_entrypoint(manifest=bad_manifest, responses_path=responses)
    assert exc.value.code is DistilleryErrorCode.MODEL_REVISION_UNPINNED


def test_validation_path_cannot_bypass_frozen_teacher_config_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _write_records(tmp_path / "responses.jsonl")
    calls: list[str] = []

    def rejecting_guard(*args: object, **kwargs: object) -> None:
        calls.append("called")
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "synthetic unsafe teacher configuration",
            )
        )

    monkeypatch.setattr(
        "distillery.training.entrypoint.assert_teacher_load_plan_frozen",
        rejecting_guard,
    )
    with pytest.raises(DistilleryError, match="unsafe teacher"):
        run_entrypoint(
            manifest=_manifest(requested="logit.v1", resolved="logit.v1"),
            responses_path=responses,
        )
    assert calls == ["called"]
