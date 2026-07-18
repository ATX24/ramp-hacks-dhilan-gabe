"""Training entrypoint: validate-only by default; requires --execute for training.

This revision never downloads weights, instantiates pretrained models, or runs
optimizer steps. ``--execute`` plus an exact acknowledgement reaches an
unimplemented hard stop before model loading or artifact writes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from distillery.contracts.errors import DistilleryError, DistilleryErrorCode, ErrorPayload
from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.contracts.manifest import SealedRunManifest
from distillery.contracts.recipes import AutoResolverInput, RecipeId
from distillery.recipes.auto import require_trainable_resolution, resolve_recipe
from distillery.recipes.base import (
    RecipeContext,
    ResponseRecord,
    require_pinned_revision,
)
from distillery.recipes.logit_v1 import LogitV1Config, LogitV1Recipe
from distillery.recipes.sequence_v1 import SequenceV1Config, SequenceV1Recipe
from distillery.training.artifacts import (
    build_run_artifact_layout,
    generate_load_instructions,
    materialization_sidecar,
)
from distillery.training.batching import (
    DEFAULT_FINANCE_MIXTURE,
    SamplerExample,
    plan_batches,
)
from distillery.training.models import (
    TrainingLoadPlan,
    assert_teacher_load_plan_frozen,
    build_training_load_plan,
)
from distillery.training.qlora import qlora_from_manifest_dict

CAPABILITY_EVIDENCE_KEY = "capability_evidence"
RESPONSE_FILE_EVIDENCE_KEY = "completion_evidence"
EXECUTE_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_TRAINING_EXECUTION"


class SpecialTokenMapEvidence(BaseModel):
    """Strict local view of the manifest's sealed special-token evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    teacher: dict[str, int]
    student: dict[str, int]


class TrainingCapabilityEvidence(BaseModel):
    """Embedded evidence used to verify auto/logit claims without running models."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["distillery.training_capabilities.v1"] = (
        "distillery.training_capabilities.v1"
    )
    special_token_maps: SpecialTokenMapEvidence | None = None
    memory_dry_run: dict[str, Any] | None = None
    auto_resolver_input: AutoResolverInput | None = None


class ResponseFileEvidence(BaseModel):
    """Manifest-bound hashes for canonical response records and JSONL bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["distillery.completion_evidence.v1"] = (
        "distillery.completion_evidence.v1"
    )
    source_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_sha256: dict[str, str]
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completion_token_counts: dict[str, int]
    completion_tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completion_token_count_source: Literal["student_tokenizer"] = (
        "student_tokenizer"
    )
    label_source_counts: dict[str, int]
    accepted_example_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_ordered_records(self) -> ResponseFileEvidence:
        keys = set(self.record_sha256)
        if not keys or keys != set(self.completion_token_counts):
            raise ValueError(
                "record hashes and completion token counts must have identical keys"
            )
        if len(keys) != self.accepted_example_count:
            raise ValueError(
                "accepted_example_count must equal sealed record count"
            )
        if any(
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in self.record_sha256.values()
        ):
            raise ValueError("record_sha256 contains malformed hashes")
        if any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for count in self.completion_token_counts.values()
        ):
            raise ValueError(
                "completion_token_counts must contain positive integers"
            )
        if sum(self.label_source_counts.values()) != self.accepted_example_count:
            raise ValueError(
                "label_source_counts must sum to accepted_example_count"
            )
        return self


@dataclass(frozen=True, slots=True)
class LoadedResponseFile:
    records: tuple[ResponseRecord, ...]
    raw_bytes: bytes


def _json_validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {
            "type": error["type"],
            "loc": list(error["loc"]),
            "msg": error["msg"],
            "input": repr(error.get("input")),
        }
        for error in exc.errors(include_url=False)
    ]


@dataclass(frozen=True, slots=True)
class EntrypointResult:
    """Outcome of a validate-only or execute-gated entrypoint invocation."""

    mode: str
    run_id: str
    recipe: str
    manifest_sha256: str
    sampler_order_hash: str
    load_plan: dict[str, Any]
    materialization: dict[str, Any] | None
    artifact_layout: dict[str, Any]
    load_instructions: str
    executed: bool


def load_manifest(path: Path) -> SealedRunManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SealedRunManifest.model_validate(raw)


def canonical_response_jsonl(records: Sequence[ResponseRecord]) -> bytes:
    """Serialize sealed records to the only accepted canonical JSONL encoding."""
    lines = [
        json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for record in records
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def canonical_records_sha256(records: Sequence[ResponseRecord]) -> str:
    return content_sha256(
        {
            "schema_version": "distillery.response_records.v1",
            "records": [
                {
                    "example_id": record.example_id,
                    "record_sha256": record.record_sha256,
                }
                for record in records
            ],
        }
    )


def provenance_records_sha256(records: Sequence[ResponseRecord]) -> str:
    return content_sha256(
        {
            "schema_version": "distillery.response_provenance.v1",
            "records": [
                {
                    "example_id": record.example_id,
                    "provenance": record.provenance_payload(),
                }
                for record in records
            ],
        }
    )


def _build_response_file_evidence_unchecked(
    records: Sequence[ResponseRecord],
) -> ResponseFileEvidence:
    if not records:
        raise ValueError("response file evidence requires at least one record")
    for record in records:
        record.assert_integrity()
    tokenizer_hashes = {
        record.completion_tokenizer_sha256 for record in records
    }
    if len(tokenizer_hashes) != 1:
        raise ValueError("all response records must use one student tokenizer")
    label_source_counts: dict[str, int] = {}
    for record in records:
        source = record.label_source.value
        label_source_counts[source] = label_source_counts.get(source, 0) + 1
    return ResponseFileEvidence(
        source_file_sha256=sha256_hex(canonical_response_jsonl(records)),
        canonical_records_sha256=canonical_records_sha256(records),
        record_sha256={
            record.example_id: record.record_sha256 for record in records
        },
        provenance_sha256=provenance_records_sha256(records),
        completion_token_counts={
            record.example_id: record.completion_token_count
            for record in records
        },
        completion_tokenizer_sha256=next(iter(tokenizer_hashes)),
        label_source_counts=label_source_counts,
        accepted_example_count=len(records),
    )


def build_response_file_evidence(
    records: Sequence[ResponseRecord],
    *,
    max_completion: int,
    max_length: int,
) -> ResponseFileEvidence:
    """Build a file seal only after enforcing independent token length caps."""
    if (
        isinstance(max_completion, bool)
        or not isinstance(max_completion, int)
        or max_completion < 1
    ):
        raise ValueError("max_completion must be an integer >= 1")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < 1
    ):
        raise ValueError("max_length must be an integer >= 1")
    completion_violations = [
        record.example_id
        for record in records
        if record.completion_token_count > max_completion
    ]
    if completion_violations:
        raise ValueError(
            "cannot seal records whose tokenizer-derived completion count "
            f"exceeds max_completion: {completion_violations}"
        )
    total_violations = [
        record.example_id
        for record in records
        if record.total_token_count > max_length
    ]
    if total_violations:
        raise ValueError(
            "cannot seal records whose joint token count exceeds max_length: "
            f"{total_violations}"
        )
    return _build_response_file_evidence_unchecked(records)


def _manifest_payload_without_capability_evidence(
    manifest: SealedRunManifest,
) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json", warnings=False)
    training = dict(payload["training"])
    qlora = dict(training["qlora"])
    qlora.pop(CAPABILITY_EVIDENCE_KEY, None)
    training["qlora"] = qlora
    payload["training"] = training
    return payload


def training_configuration_sha256(manifest: SealedRunManifest) -> str:
    """Hash the exact training configuration, excluding its evidence envelope."""
    payload = _manifest_payload_without_capability_evidence(manifest)
    return content_sha256(payload["training"])


def model_configuration_sha256(model_spec: Any) -> str:
    """Hash model identity plus tokenizer/chat-template configuration."""
    return content_sha256(model_spec.model_dump(mode="json"))


def length_configuration_sha256(manifest: SealedRunManifest) -> str:
    return content_sha256(
        {
            "max_length": manifest.training.max_length,
            "max_completion": int(
                manifest.training.qlora.get("max_completion", 160)
            ),
            "vocab_chunk_size": int(
                manifest.training.qlora.get("vocab_chunk", 4096)
            ),
        }
    )


def capability_binding_sha256(
    manifest: SealedRunManifest,
    *,
    teacher_special_token_map: dict[str, int],
    student_special_token_map: dict[str, int],
    auto_resolver_input: dict[str, Any] | None = None,
) -> str:
    """
    Bind precomputed evidence to the exact sealed content except the evidence itself.

    The special-token maps are included explicitly so evidence cannot be moved
    between otherwise-identical manifests with different token semantics.
    """
    return content_sha256(
        {
            "manifest_without_capability_evidence": (
                _manifest_payload_without_capability_evidence(manifest)
            ),
            "special_token_maps": {
                "teacher": teacher_special_token_map,
                "student": student_special_token_map,
            },
            "auto_resolver_input": auto_resolver_input,
        }
    )


def parse_capability_evidence(
    manifest: SealedRunManifest,
) -> TrainingCapabilityEvidence:
    raw = manifest.training.qlora.get(CAPABILITY_EVIDENCE_KEY)
    if raw is None:
        return TrainingCapabilityEvidence()
    try:
        return TrainingCapabilityEvidence.model_validate(raw)
    except ValidationError as exc:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "embedded training capability evidence is malformed",
                details={
                    "validation_errors": _json_validation_errors(exc),
                },
                run_id=manifest.run_id,
            )
        ) from None


def parse_response_file_evidence(
    manifest: SealedRunManifest,
) -> ResponseFileEvidence:
    raw = manifest.training.completion_evidence
    if raw is None:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "manifest lacks sealed response file evidence",
                details={
                    "missing": RESPONSE_FILE_EVIDENCE_KEY,
                    "required_manifest_field": "training.completion_evidence",
                },
                run_id=manifest.run_id,
            )
        )
    try:
        payload = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
        return ResponseFileEvidence.model_validate(payload)
    except ValidationError as exc:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "sealed response file evidence is malformed",
                details={"validation_errors": _json_validation_errors(exc)},
                run_id=manifest.run_id,
            )
        ) from None


def validate_response_file_evidence(
    loaded: LoadedResponseFile,
    expected: ResponseFileEvidence,
    *,
    run_id: str,
) -> None:
    canonical_bytes = canonical_response_jsonl(loaded.records)
    actual = _build_response_file_evidence_unchecked(loaded.records)
    violations: dict[str, Any] = {}
    if loaded.raw_bytes != canonical_bytes:
        violations["canonical_jsonl"] = {
            "expected": "sorted compact UTF-8 JSONL with one trailing newline",
            "actual_file_sha256": sha256_hex(loaded.raw_bytes),
        }
    for field_name in (
        "source_file_sha256",
        "canonical_records_sha256",
        "record_sha256",
        "provenance_sha256",
        "completion_token_counts",
        "completion_tokenizer_sha256",
        "completion_token_count_source",
        "label_source_counts",
        "accepted_example_count",
    ):
        expected_value = getattr(expected, field_name)
        actual_value = getattr(actual, field_name)
        if expected_value != actual_value:
            violations[field_name] = {
                "expected": expected_value,
                "actual": actual_value,
            }
    if violations:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "response file content or provenance does not match sealed evidence",
                details={
                    "violations": violations,
                    "content_and_provenance_bound": True,
                },
                run_id=run_id,
            )
        )


def validate_manifest_structure(manifest: SealedRunManifest) -> tuple[str, ...]:
    """Pure structural checks; returns warnings (errors raise DistilleryError)."""
    warnings: list[str] = []
    if manifest.training.seed < 0:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "training.seed must be non-negative",
                details={"seed": manifest.training.seed},
                run_id=manifest.run_id,
            )
        )
    if manifest.recipe.resolved not in {
        RecipeId.SEQUENCE_V1.value,
        RecipeId.LOGIT_V1.value,
        "do_not_distill",
    }:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_NOT_IMPLEMENTED,
                f"unsupported resolved recipe {manifest.recipe.resolved}",
                details={"resolved": manifest.recipe.resolved},
                run_id=manifest.run_id,
            )
        )
    if manifest.recipe.resolved == "do_not_distill":
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.CAPABILITY_UNAVAILABLE,
                "manifest resolved to do_not_distill; refusing training entrypoint",
                run_id=manifest.run_id,
            )
        )
    require_pinned_revision(
        manifest.models.teacher.revision,
        role="teacher",
        run_id=manifest.run_id,
    )
    require_pinned_revision(
        manifest.models.student.revision,
        role="student",
        run_id=manifest.run_id,
    )
    sealed = manifest.seal_sha256()
    if len(sealed) != 64:
        warnings.append("unexpected_manifest_hash_length")
    return tuple(warnings)


def build_recipe_context(
    manifest: SealedRunManifest,
    evidence: TrainingCapabilityEvidence,
) -> RecipeContext:
    max_completion = int(manifest.training.qlora.get("max_completion", 160))
    special_tokens = evidence.special_token_maps
    teacher_special_tokens = special_tokens.teacher if special_tokens else {}
    student_special_tokens = special_tokens.student if special_tokens else {}
    return RecipeContext(
        run_id=manifest.run_id,
        seed=manifest.training.seed,
        max_length=manifest.training.max_length,
        max_completion=max_completion,
        student_model_id=manifest.models.student.id,
        student_revision=manifest.models.student.revision,
        teacher_model_id=manifest.models.teacher.id,
        teacher_revision=manifest.models.teacher.revision,
        tokenizer_sha256_student=manifest.models.student.tokenizer_sha256,
        tokenizer_sha256_teacher=manifest.models.teacher.tokenizer_sha256,
        chat_template_sha256_student=manifest.models.student.chat_template_sha256,
        chat_template_sha256_teacher=manifest.models.teacher.chat_template_sha256,
        special_token_map_student=student_special_tokens,
        special_token_map_teacher=teacher_special_tokens,
        memory_dry_run_evidence=evidence.memory_dry_run,
        capability_binding_sha256=capability_binding_sha256(
            manifest,
            teacher_special_token_map=teacher_special_tokens,
            student_special_token_map=student_special_tokens,
            auto_resolver_input=(
                evidence.auto_resolver_input.model_dump(mode="json")
                if evidence.auto_resolver_input is not None
                else None
            ),
        ),
        training_config_sha256=training_configuration_sha256(manifest),
        teacher_model_config_sha256=model_configuration_sha256(
            manifest.models.teacher
        ),
        student_model_config_sha256=model_configuration_sha256(
            manifest.models.student
        ),
        length_config_sha256=length_configuration_sha256(manifest),
        runtime_image_digest=manifest.runtime.image_digest,
        runtime_instance_type=manifest.runtime.instance_type,
        extras={
            "capability_evidence_adapter": (
                f"training.qlora.{CAPABILITY_EVIDENCE_KEY}"
            )
        },
    )


def select_recipe(
    manifest: SealedRunManifest,
    evidence: TrainingCapabilityEvidence,
) -> SequenceV1Recipe | LogitV1Recipe:
    resolved = manifest.recipe.resolved
    if manifest.recipe.requested == "auto":
        if evidence.auto_resolver_input is None:
            raise DistilleryError(
                ErrorPayload.from_code(
                    DistilleryErrorCode.AUTO_RESOLVER_FAILED,
                    "auto-resolved manifest lacks sealed resolver input evidence",
                    details={
                        "missing": "auto_resolver_input",
                        "manifest_resolved": resolved,
                        "integration_requirement": (
                            "add sealed AutoResolverInput to the foundation manifest"
                        ),
                    },
                    run_id=manifest.run_id,
                )
            )
        record = resolve_recipe(
            RecipeId.AUTO,
            auto_input=evidence.auto_resolver_input,
        )
        recomputed = require_trainable_resolution(record)
        if recomputed != resolved or tuple(record.reasons) != tuple(
            manifest.recipe.resolver_reasons
        ):
            raise DistilleryError(
                ErrorPayload.from_code(
                    DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                    "sealed auto resolution does not match embedded resolver evidence",
                    details={
                        "manifest_resolved": resolved,
                        "recomputed_resolved": recomputed,
                        "manifest_reasons": list(manifest.recipe.resolver_reasons),
                        "recomputed_reasons": list(record.reasons),
                    },
                    run_id=manifest.run_id,
                )
            )
    else:
        record = resolve_recipe(manifest.recipe.requested)
        trainable = require_trainable_resolution(record)
        if trainable != resolved:
            raise DistilleryError(
                ErrorPayload.from_code(
                    DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                    "manifest resolved recipe disagrees with requested recipe",
                    details={
                        "requested": manifest.recipe.requested,
                        "manifest_resolved": resolved,
                        "recomputed": trainable,
                    },
                    run_id=manifest.run_id,
                )
            )
    if resolved == RecipeId.SEQUENCE_V1.value:
        return SequenceV1Recipe(
            SequenceV1Config(
                max_completion=int(
                    manifest.training.qlora.get("max_completion", 160)
                ),
                max_length=manifest.training.max_length,
            )
        )
    if resolved == RecipeId.LOGIT_V1.value:
        qlora = manifest.training.qlora
        cfg = LogitV1Config(
            temperature=float(qlora.get("logit_temperature", 2.0)),
            kd_weight=float(qlora.get("kd_weight", 0.7)),
            hard_ce_weight=float(qlora.get("hard_ce_weight", 0.3)),
            vocab_chunk_size=int(qlora.get("vocab_chunk", 4096)),
            max_completion=int(qlora.get("max_completion", 160)),
        )
        return LogitV1Recipe(cfg)
    raise DistilleryError(
        ErrorPayload.from_code(
            DistilleryErrorCode.RECIPE_NOT_IMPLEMENTED,
            f"no trainer for recipe {resolved}",
            run_id=manifest.run_id,
        )
    )


def build_load_plan_from_manifest(manifest: SealedRunManifest) -> TrainingLoadPlan:
    return build_training_load_plan(
        recipe=manifest.recipe.resolved,
        seed=manifest.training.seed,
        student_id=manifest.models.student.id,
        student_revision=manifest.models.student.revision,
        teacher_id=manifest.models.teacher.id,
        teacher_revision=manifest.models.teacher.revision,
        qlora=dict(manifest.training.qlora),
        student_tokenizer_sha256=manifest.models.student.tokenizer_sha256,
        teacher_tokenizer_sha256=manifest.models.teacher.tokenizer_sha256,
        student_chat_template_sha256=manifest.models.student.chat_template_sha256,
        teacher_chat_template_sha256=manifest.models.teacher.chat_template_sha256,
    )


def _load_response_records(path: Path | None) -> LoadedResponseFile:
    if path is None:
        return LoadedResponseFile(records=(), raw_bytes=b"")
    raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "response file must be valid UTF-8",
                details={"reason": str(exc)},
            )
        ) from None
    records: list[ResponseRecord] = []
    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(ResponseRecord.model_validate_json(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            details: dict[str, Any] = {"line": line_number}
            if isinstance(exc, ValidationError):
                details["validation_errors"] = _json_validation_errors(exc)
            else:
                details["reason"] = str(exc)
            raise DistilleryError(
                ErrorPayload.from_code(
                    DistilleryErrorCode.INVALID_DATASET,
                    "response/token evidence line is invalid",
                    details=details,
                )
            ) from None
    return LoadedResponseFile(records=tuple(records), raw_bytes=raw_bytes)


def run_entrypoint(
    *,
    manifest: SealedRunManifest,
    responses_path: Path | None = None,
    execute: bool = False,
    execute_acknowledgement: str | None = None,
    output_dir: Path | None = None,
) -> EntrypointResult:
    """
    Validate a sealed manifest and prepare training artifacts.

    Default ``execute=False`` is validation-only: no model download, no optimizer,
    no SageMaker. Setting ``execute=True`` is required before any training path;
    this revision still refuses to pull weights before writing any sidecar.
    """
    validate_manifest_structure(manifest)
    qlora_cfg = qlora_from_manifest_dict(dict(manifest.training.qlora))
    evidence = parse_capability_evidence(manifest)
    recipe = select_recipe(manifest, evidence)
    context = build_recipe_context(manifest, evidence)
    recipe.validate_capabilities(context)

    load_plan = build_load_plan_from_manifest(manifest)
    # Config-level proof is mandatory on validation and execute paths.
    assert_teacher_load_plan_frozen(load_plan, run_id=manifest.run_id)

    if execute:
        if execute_acknowledgement != EXECUTE_ACKNOWLEDGEMENT:
            raise DistilleryError(
                ErrorPayload.from_code(
                    DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                    "training execution requires the exact explicit acknowledgement",
                    details={
                        "required_acknowledgement": EXECUTE_ACKNOWLEDGEMENT,
                        "environment_variables_accepted": False,
                    },
                    run_id=manifest.run_id,
                )
            )
        # Hard stop occurs before materialization/output writes/model loading.
        # A future implementation must next call freeze_and_assert_runtime_teacher
        # and construct_optimizer_after_teacher_guard from training.models.
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.CAPABILITY_UNAVAILABLE,
                "training execute path is unimplemented and hard-gated; "
                "no model or optimizer was created",
                details={
                    "run_id": manifest.run_id,
                    "recipe": manifest.recipe.resolved,
                    "sidecars_written": False,
                    "next_required_runtime_guard": (
                        "freeze_and_assert_runtime_teacher before "
                        "construct_optimizer_after_teacher_guard"
                    ),
                },
                run_id=manifest.run_id,
            )
        )
    if execute_acknowledgement is not None:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "execute acknowledgement is invalid without execute=True",
                run_id=manifest.run_id,
            )
        )
    if responses_path is None:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "response/token evidence is required to verify sampler_order_hash",
                details={
                    "missing": "responses_path",
                    "reason": "sealed sampler hash cannot be recomputed without examples",
                },
                run_id=manifest.run_id,
            )
        )

    response_file_evidence = parse_response_file_evidence(manifest)
    loaded_responses = _load_response_records(responses_path)
    records = loaded_responses.records
    if not records:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "response/token evidence contains zero examples",
                run_id=manifest.run_id,
            )
        )
    validate_response_file_evidence(
        loaded_responses,
        response_file_evidence,
        run_id=manifest.run_id,
    )
    tokenizer_mismatches = [
        record.example_id
        for record in records
        if record.completion_tokenizer_sha256
        != manifest.models.student.tokenizer_sha256
    ]
    if tokenizer_mismatches:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "record tokenization evidence uses the wrong student tokenizer",
                details={
                    "example_ids": tokenizer_mismatches,
                    "expected_student_tokenizer_sha256": (
                        manifest.models.student.tokenizer_sha256
                    ),
                },
                run_id=manifest.run_id,
            )
        )
    completion_cap_violations = [
        {
            "example_id": record.example_id,
            "completion_token_count": record.completion_token_count,
            "max_completion": context.max_completion,
        }
        for record in records
        if record.completion_token_count > context.max_completion
    ]
    if completion_cap_violations:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "tokenizer-derived completion count exceeds max_completion",
                details={"violations": completion_cap_violations},
                run_id=manifest.run_id,
            )
        )
    total_length_violations = [
        {
            "example_id": record.example_id,
            "total_token_count": record.total_token_count,
            "max_length": context.max_length,
        }
        for record in records
        if record.total_token_count > context.max_length
    ]
    if total_length_violations:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "joint tokenizer-derived total count exceeds max_length",
                details={"violations": total_length_violations},
                run_id=manifest.run_id,
            )
        )
    report = recipe.materialize(records, context=context)
    sealed_ids = set(response_file_evidence.record_sha256)
    accepted_ids = {example.example_id for example in report.accepted}
    if accepted_ids != sealed_ids:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "sealed completion evidence must contain exactly accepted examples",
                details={
                    "missing_from_materialization": sorted(
                        sealed_ids - accepted_ids
                    ),
                    "unsealed_accepted_examples": sorted(
                        accepted_ids - sealed_ids
                    ),
                    "rejected_example_ids": [
                        example.example_id for example in report.rejected
                    ],
                },
                run_id=manifest.run_id,
            )
        )
    sampler_examples = [
        SamplerExample(
            example_id=ex.example_id,
            task=ex.task,
            difficulty=ex.difficulty,
            completion_tokens=ex.completion_token_count,
            prompt_tokens=ex.tokenization.prompt_token_count,
            total_tokens=ex.total_token_count,
            completion_token_source=ex.completion_token_count_source,
            completion_tokenizer_sha256=ex.completion_tokenizer_sha256,
            record_sha256=ex.record_sha256,
        )
        for ex in report.accepted
    ]
    try:
        plan = plan_batches(
            sampler_examples,
            seed=manifest.training.seed,
            microbatch_size=1,
            mixture=DEFAULT_FINANCE_MIXTURE,
        )
    except ValueError as exc:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "deterministic sampler validation failed",
                details={"reason": str(exc)},
                run_id=manifest.run_id,
            )
        ) from None
    if plan.sampler_order_hash != manifest.sampler_order_hash:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_DATASET,
                "sealed sampler_order_hash does not match recomputed order",
                details={
                    "sealed_sampler_order_hash": manifest.sampler_order_hash,
                    "recomputed_sampler_order_hash": plan.sampler_order_hash,
                    "sealed_value_preserved": True,
                },
                run_id=manifest.run_id,
            )
        )
    order_hash = manifest.sampler_order_hash
    materialization_payload = materialization_sidecar(
        accepted_example_ids=[ex.example_id for ex in report.accepted],
        rejected_example_ids=[ex.example_id for ex in report.rejected],
        label_source_counts=report.label_source_counts,
        recipe_id=manifest.recipe.resolved,
        sampler_order_hash=order_hash,
        completion_token_counts={
            ex.example_id: ex.completion_token_count
            for ex in report.accepted
        },
        completion_tokenizer_sha256=manifest.models.student.tokenizer_sha256,
        canonical_records_sha256=response_file_evidence.canonical_records_sha256,
        source_file_sha256=response_file_evidence.source_file_sha256,
        provenance_sha256=response_file_evidence.provenance_sha256,
        record_sha256={
            ex.example_id: ex.record_sha256 for ex in report.accepted
        },
    )

    layout = build_run_artifact_layout(
        run_id=manifest.run_id,
        root_prefix=manifest.output.prefix,
    )
    load_instructions = generate_load_instructions(
        student_base_id=manifest.models.student.id,
        student_revision=manifest.models.student.revision,
        adapter_uri=layout.resolve("model_adapter_dir"),
        merged_uri=layout.resolve("model_merged_dir"),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_result.json").write_text(
            json.dumps(
                {
                    "run_id": manifest.run_id,
                    "recipe": manifest.recipe.resolved,
                    "manifest_sha256": manifest.seal_sha256(),
                    "sampler_order_hash": order_hash,
                    "qlora": qlora_cfg.model_dump(mode="json"),
                    "mode": "validation_only",
                    "executed": False,
                    "execution_gate": "unimplemented_hard_stop",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "materialization.json").write_text(
            json.dumps(materialization_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "load_instructions.py.txt").write_text(
            load_instructions, encoding="utf-8"
        )

    return EntrypointResult(
        mode="validate_only",
        run_id=manifest.run_id,
        recipe=manifest.recipe.resolved,
        manifest_sha256=manifest.seal_sha256(),
        sampler_order_hash=order_hash,
        load_plan=load_plan.model_dump(mode="json"),
        materialization=materialization_payload,
        artifact_layout={
            "root_prefix": layout.root_prefix,
            "files": [f.model_dump(mode="json") for f in layout.files],
        },
        load_instructions=load_instructions,
        executed=False,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="distillery-train",
        description=(
            "Distillery trainer entrypoint. Defaults to validation-only. "
            "Pass --execute to enter the training path (explicit opt-in)."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to sealed distillery.run.v1 manifest JSON",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="JSONL ResponseRecord/token evidence used to recompute sampler hash",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for validation sidecars",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Required opt-in before any training path (downloads/optimizer)",
    )
    parser.add_argument(
        "--execute-acknowledgement",
        default=None,
        help=(
            "Exact acknowledgement required with --execute; environment variables "
            "are intentionally ignored"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Force validation-only (default behavior; mutually documents intent)",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = stdout or sys.stdout
    if args.execute and args.validate_only:
        print(
            json.dumps(
                {
                    "error": "cannot combine --execute with --validate-only",
                    "code": DistilleryErrorCode.RECIPE_INCOMPATIBLE.value,
                }
            ),
            file=out,
        )
        return 2

    execute = bool(args.execute) and not bool(args.validate_only)
    try:
        manifest = load_manifest(args.manifest)
        result = run_entrypoint(
            manifest=manifest,
            responses_path=args.responses,
            execute=execute,
            execute_acknowledgement=args.execute_acknowledgement,
            output_dir=args.output_dir,
        )
    except DistilleryError as exc:
        print(
            json.dumps(exc.payload.model_dump(mode="json"), sort_keys=True),
            file=out,
        )
        return 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "code": DistilleryErrorCode.INVALID_DATASET.value,
                    "message": str(exc),
                    "retryable": False,
                },
                sort_keys=True,
            ),
            file=out,
        )
        return 1

    print(
        json.dumps(
            {
                "mode": result.mode,
                "run_id": result.run_id,
                "recipe": result.recipe,
                "manifest_sha256": result.manifest_sha256,
                "sampler_order_hash": result.sampler_order_hash,
                "executed": result.executed,
            },
            sort_keys=True,
        ),
        file=out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
