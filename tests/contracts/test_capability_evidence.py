"""Sealed logit capability and completion-provenance evidence."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from distillery.contracts.hashing import content_sha256
from distillery.contracts.manifest import (
    ManifestCompletionEvidence,
    ManifestQLoRAConfig,
    ManifestSpecialTokenMapEvidence,
    ManifestTraining,
    SealedRunManifest,
    manifest_capability_binding_sha256,
    manifest_memory_dry_run_evidence_sha256,
    manifest_training_configuration_sha256,
)
from distillery.contracts.recipes import (
    AUTO_BASELINE_PRECEDENCE_REASON,
    AUTO_LOGIT_REASONS,
    AutoResolverInput,
)
from distillery.contracts.tasks import LabelSource
from distillery.training.artifacts import materialization_sidecar
from distillery.training.entrypoint import (
    TrainingCapabilityEvidence,
    build_recipe_context,
    capability_binding_sha256,
    parse_capability_evidence,
    parse_response_file_evidence,
    select_recipe,
    training_configuration_sha256,
    validate_manifest_structure,
)

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
REV_A = "a" * 40
REV_B = "b" * 40


def _capability_input(**updates: object) -> AutoResolverInput:
    payload: dict[str, object] = {
        "local_white_box": True,
        "tokenizer_fingerprint_match": True,
        "special_token_map_match": True,
        "chat_template_compatible": True,
        "memory_dry_run_ok": True,
    }
    payload.update(updates)
    return AutoResolverInput.model_validate(payload)


def _payload(*, requested: str = "auto") -> dict[str, object]:
    auto_input = _capability_input()
    special_tokens = ManifestSpecialTokenMapEvidence(
        teacher={"bos_token_id": 1, "eos_token_id": 2},
        student={"bos_token_id": 1, "eos_token_id": 2},
    )
    training = ManifestTraining(
        seed=17,
        max_steps=30,
        token_budget=1_000,
        max_length=512,
        qlora=ManifestQLoRAConfig(),
        completion_evidence=ManifestCompletionEvidence(
            source_file_sha256=HEX_B,
            canonical_records_sha256=HEX_A,
            record_sha256={"ex_training_001": HEX_C},
            provenance_sha256=HEX_B,
            completion_token_counts={"ex_training_001": 24},
            completion_tokenizer_sha256=HEX_A,
            label_source_counts={LabelSource.ORACLE: 1},
            accepted_example_count=1,
        ),
    )
    training_payload = training.model_dump(mode="json")
    training_payload["qlora"]["capability_evidence"] = {
        "schema_version": "distillery.training_capabilities.v1",
        "special_token_maps": special_tokens.model_dump(mode="json"),
        "memory_dry_run": {
            "schema_version": "distillery.memory_dry_run.v2",
            "passed": True,
            "binding_sha256": "0" * 64,
            "evidence_sha256": "0" * 64,
            "training_config_sha256": "0" * 64,
            "teacher_model_config_sha256": "0" * 64,
            "student_model_config_sha256": "0" * 64,
            "length_config_sha256": "0" * 64,
            "runtime_image_digest": f"sha256:{HEX_C}",
            "instance_type": "ml.g5.xlarge",
            "recipe_id": "logit.v1",
            "teacher_model_id": "teacher/model",
            "teacher_revision": REV_A,
            "student_model_id": "student/model",
            "student_revision": REV_B,
            "max_length": 512,
            "max_completion": 160,
            "vocab_chunk_size": 4096,
            "peak_memory_bytes": 8_000_000_000,
            "capacity_memory_bytes": 24_000_000_000,
            "headroom_bytes": 16_000_000_000,
            "device_type": "NVIDIA A10G",
            "probe_id": "probe-a10g-001",
        },
        "auto_resolver_input": auto_input.model_dump(mode="json"),
    }
    payload: dict[str, object] = {
        "schema_version": "distillery.run.v1",
        "run_id": "run_capability_001",
        "created_at": "2026-07-18T16:00:00Z",
        "dataset": {
            "dataset_id": "ds_finance_world_v1",
            "uri": "s3://bucket/dataset/",
            "sha256": HEX_A,
            "split_sha256": {"train": HEX_A, "validation": HEX_B},
        },
        "models": {
            "teacher": {
                "id": "teacher/model",
                "revision": REV_A,
                "tokenizer_sha256": HEX_A,
                "chat_template_sha256": HEX_B,
            },
            "student": {
                "id": "student/model",
                "revision": REV_B,
                "tokenizer_sha256": HEX_A,
                "chat_template_sha256": HEX_B,
            },
        },
        "recipe": {
            "requested": requested,
            "resolved": "logit.v1",
            "resolver_reasons": (
                AUTO_LOGIT_REASONS if requested == "auto" else ("explicit_request",)
            ),
        },
        "training": training_payload,
        "proof_protocol": {"id": "finance-proof.v1", "sha256": HEX_C},
        "runtime": {
            "backend": "sagemaker",
            "region": "us-east-1",
            "instance_type": "ml.g5.xlarge",
            "image_digest": f"sha256:{HEX_C}",
        },
        "cost": {
            "max_run_usd": 25.0,
            "estimate_low_usd": 2.0,
            "estimate_high_usd": 5.0,
        },
        "output": {"prefix": "s3://bucket/runs/run_capability_001/"},
        "package_lock_hash": HEX_C,
        "source_revision": "contracts-v1",
        "license_dispositions": {},
        "tags": {},
        "sampler_order_hash": HEX_B,
    }
    _refresh_bindings(payload)
    return payload


def _refresh_bindings(payload: dict[str, object]) -> None:
    training = payload["training"]
    assert isinstance(training, dict)
    qlora = training["qlora"]
    assert isinstance(qlora, dict)
    capability = qlora["capability_evidence"]
    assert isinstance(capability, dict)
    memory = capability["memory_dry_run"]
    assert isinstance(memory, dict)
    memory["training_config_sha256"] = manifest_training_configuration_sha256(payload)
    models = payload["models"]
    runtime = payload["runtime"]
    assert isinstance(models, dict)
    assert isinstance(runtime, dict)
    memory["teacher_model_config_sha256"] = content_sha256(models["teacher"])
    memory["student_model_config_sha256"] = content_sha256(models["student"])
    memory["length_config_sha256"] = content_sha256(
        {
            "max_length": training["max_length"],
            "max_completion": qlora["max_completion"],
            "vocab_chunk_size": qlora["vocab_chunk"],
        }
    )
    memory["runtime_image_digest"] = runtime["image_digest"]
    memory["instance_type"] = runtime["instance_type"]
    memory["binding_sha256"] = manifest_capability_binding_sha256(payload)
    memory["evidence_sha256"] = manifest_memory_dry_run_evidence_sha256(memory)


@pytest.mark.parametrize("requested", ["auto", "logit.v1"])
def test_logit_requires_and_accepts_complete_bound_evidence(requested: str) -> None:
    manifest = SealedRunManifest.model_validate(_payload(requested=requested))
    assert manifest.recipe.resolved == "logit.v1"
    assert manifest.training.qlora.capability_evidence is not None


def test_unknown_logit_capability_is_not_success() -> None:
    payload = _payload(requested="logit.v1")
    capability = payload["training"]["qlora"]["capability_evidence"]  # type: ignore[index]
    capability["auto_resolver_input"]["memory_dry_run_ok"] = None  # type: ignore[index]
    _refresh_bindings(payload)
    with pytest.raises(ValidationError, match="unknown or failed"):
        SealedRunManifest.model_validate(payload)


@pytest.mark.parametrize("missing", ["all", "auto_resolver_input", "memory_dry_run"])
def test_explicit_logit_rejects_incomplete_capability_evidence(
    missing: str,
) -> None:
    payload = _payload(requested="logit.v1")
    qlora = payload["training"]["qlora"]  # type: ignore[index]
    capability = qlora["capability_evidence"]  # type: ignore[index]
    if missing == "all":
        qlora["capability_evidence"] = None  # type: ignore[index]
    else:
        capability[missing] = None  # type: ignore[index]
    with pytest.raises(ValidationError):
        SealedRunManifest.model_validate(payload)


def test_auto_resolution_is_recomputed_from_sealed_input() -> None:
    payload = _payload()
    capability = payload["training"]["qlora"]["capability_evidence"]  # type: ignore[index]
    capability["auto_resolver_input"]["usable_responses_exist"] = True  # type: ignore[index]
    _refresh_bindings(payload)
    with pytest.raises(ValidationError, match="do not match recipe resolution"):
        SealedRunManifest.model_validate(payload)


def test_do_not_distill_manifest_has_resolver_evidence_but_no_training_evidence() -> None:
    payload = _payload()
    payload["recipe"] = {
        "requested": "auto",
        "resolved": "do_not_distill",
        "resolver_reasons": [AUTO_BASELINE_PRECEDENCE_REASON],
    }
    payload["training"]["completion_evidence"] = None  # type: ignore[index]
    capability = payload["training"]["qlora"]["capability_evidence"]  # type: ignore[index]
    capability["special_token_maps"] = None  # type: ignore[index]
    capability["memory_dry_run"] = None  # type: ignore[index]
    capability["auto_resolver_input"] = AutoResolverInput(  # type: ignore[index]
        cheaper_baseline_satisfies_gate=True
    ).model_dump(mode="json")
    manifest = SealedRunManifest.model_validate(payload)
    assert manifest.recipe.resolved == "do_not_distill"
    assert manifest.training.completion_evidence is None


def test_logit_rejects_missing_or_empty_special_token_evidence() -> None:
    payload = _payload()
    capability = payload["training"]["qlora"]["capability_evidence"]  # type: ignore[index]
    capability["special_token_maps"]["teacher"] = {}  # type: ignore[index]
    _refresh_bindings(payload)
    with pytest.raises(ValidationError):
        SealedRunManifest.model_validate(payload)


def test_memory_evidence_self_digest_and_capacity_are_validated() -> None:
    payload = _payload()
    memory = payload["training"]["qlora"]["capability_evidence"][  # type: ignore[index]
        "memory_dry_run"
    ]
    memory["probe_id"] = "tampered-probe"  # type: ignore[index]
    with pytest.raises(ValidationError, match="evidence_sha256"):
        SealedRunManifest.model_validate(payload)

    payload = _payload()
    memory = payload["training"]["qlora"]["capability_evidence"][  # type: ignore[index]
        "memory_dry_run"
    ]
    memory["headroom_bytes"] = 1  # type: ignore[index]
    memory["evidence_sha256"] = manifest_memory_dry_run_evidence_sha256(memory)  # type: ignore[arg-type,index]
    with pytest.raises(ValidationError, match="headroom_bytes"):
        SealedRunManifest.model_validate(payload)


@pytest.mark.parametrize("binding_target", ["model", "config", "image"])
def test_memory_probe_is_bound_to_model_config_and_image(
    binding_target: str,
) -> None:
    payload = _payload()
    if binding_target == "model":
        payload["models"]["teacher"]["id"] = "other/teacher"  # type: ignore[index]
    elif binding_target == "config":
        payload["training"]["qlora"]["vocab_chunk"] = 2048  # type: ignore[index]
    else:
        payload["runtime"]["image_digest"] = f"sha256:{HEX_B}"  # type: ignore[index]
    with pytest.raises(ValidationError, match="binding mismatch"):
        SealedRunManifest.model_validate(payload)


def test_completion_counts_are_required_and_student_tokenizer_bound() -> None:
    payload = _payload()
    payload["training"]["completion_evidence"] = None  # type: ignore[index]
    with pytest.raises(ValidationError, match="completion evidence"):
        SealedRunManifest.model_validate(payload)

    payload = _payload()
    completion = payload["training"]["completion_evidence"]  # type: ignore[index]
    completion["completion_tokenizer_sha256"] = HEX_B  # type: ignore[index]
    _refresh_bindings(payload)
    with pytest.raises(ValidationError, match="student tokenizer"):
        SealedRunManifest.model_validate(payload)


def test_current_training_adapter_reads_identical_typed_evidence() -> None:
    manifest = SealedRunManifest.model_validate(_payload())
    assert validate_manifest_structure(manifest) == ()
    raw = manifest.training.qlora.get("capability_evidence")
    adapter_evidence = TrainingCapabilityEvidence.model_validate(raw)
    assert adapter_evidence.special_token_maps is not None
    assert training_configuration_sha256(manifest) == manifest_training_configuration_sha256(
        manifest
    )
    assert capability_binding_sha256(
        manifest,
        teacher_special_token_map=dict(adapter_evidence.special_token_maps.teacher),
        student_special_token_map=dict(adapter_evidence.special_token_maps.student),
        auto_resolver_input=(
            adapter_evidence.auto_resolver_input.model_dump(mode="json")
            if adapter_evidence.auto_resolver_input is not None
            else None
        ),
    ) == manifest_capability_binding_sha256(manifest)
    completion = manifest.training.completion_evidence
    assert completion is not None
    sidecar = materialization_sidecar(
        accepted_example_ids=completion.completion_token_counts,
        rejected_example_ids=(),
        label_source_counts={
            source.value: count for source, count in completion.label_source_counts.items()
        },
        recipe_id=manifest.recipe.resolved,
        sampler_order_hash=manifest.sampler_order_hash,
        completion_token_counts=completion.completion_token_counts,
        completion_tokenizer_sha256=completion.completion_tokenizer_sha256,
        canonical_records_sha256=completion.canonical_records_sha256,
        source_file_sha256=completion.source_file_sha256,
        provenance_sha256=completion.provenance_sha256,
        record_sha256=completion.record_sha256,
    )
    assert sidecar["completion_token_counts"] == dict(completion.completion_token_counts)
    assert sidecar["completion_token_count_source"] == completion.completion_token_count_source
    assert sidecar["completion_tokenizer_sha256"] == completion.completion_tokenizer_sha256
    assert sidecar["canonical_records_sha256"] == completion.canonical_records_sha256
    assert sidecar["source_file_sha256"] == completion.source_file_sha256
    assert sidecar["provenance_sha256"] == completion.provenance_sha256
    assert sidecar["record_sha256"] == {
        "ex_training_001": HEX_C,
    }
    parsed_response_file = parse_response_file_evidence(manifest)
    assert parsed_response_file.model_dump(mode="json") == completion.model_dump(mode="json")
    parsed = parse_capability_evidence(manifest)
    recipe = select_recipe(manifest, parsed)
    recipe.validate_capabilities(build_recipe_context(manifest, parsed))


def test_nested_capability_evidence_is_immutable_and_seal_stable() -> None:
    manifest = SealedRunManifest.model_validate(_payload())
    before = manifest.seal_sha256()
    evidence = manifest.training.qlora.capability_evidence
    assert evidence is not None
    assert evidence.special_token_maps is not None
    with pytest.raises(TypeError):
        evidence.special_token_maps.teacher["bos_token_id"] = 99  # type: ignore[index]
    assert manifest.seal_sha256() == before


def test_capability_binding_changes_when_image_changes() -> None:
    payload = _payload()
    original = manifest_capability_binding_sha256(payload)
    changed = deepcopy(payload)
    changed["runtime"]["image_digest"] = f"sha256:{HEX_B}"  # type: ignore[index]
    assert manifest_capability_binding_sha256(changed) != original
