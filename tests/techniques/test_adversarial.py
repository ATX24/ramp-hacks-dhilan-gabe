"""Adversarial regressions through the public BYODT planning seam."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from distillery.contracts.errors import DistilleryErrorCode
from distillery.techniques import (
    CompatibilityContext,
    TechniqueDescriptor,
    TechniqueError,
    TechniqueErrorCode,
    TechniquePlan,
    TechniqueRegistry,
    TechniqueRequest,
    recompute_protocol_hash,
)
from distillery.techniques.capabilities import (
    EvidenceRequirement,
    TechniqueCapability,
)
from distillery.techniques.descriptor import (
    DESCRIPTOR_SCHEMA_VERSION,
    ArtifactContract,
    CostModel,
    ExecutionKind,
    HardwareRequirements,
    PluginImageBinding,
    ReviewedSourceBinding,
    TeacherSignal,
    TokenizerConstraint,
)
from distillery.techniques.errors import technique_error


def _external_kwargs(**overrides):
    base = {
        "technique_id": "hackathon.dhilan.reverse_kl",
        "version": "1.0.0",
        "display_name": "reverse KL",
        "summary": "plan-only external technique",
        "execution": ExecutionKind.EXTERNAL_CONTAINER,
        "teacher_signal": TeacherSignal.FULL_LOGITS,
        "tokenizer_constraint": TokenizerConstraint.EXACT_MATCH,
        "capabilities": (
            TechniqueCapability.FULL_LOGITS.value,
            TechniqueCapability.LOCAL_WHITE_BOX.value,
            TechniqueCapability.NETWORK_ISOLATED_PLUGIN.value,
            TechniqueCapability.CUSTOM_OBJECTIVE.value,
            TechniqueCapability.DETERMINISTIC_PLAN.value,
        ),
        "evidence_requirements": (
            EvidenceRequirement.PINNED_STUDENT_REVISION.value,
            EvidenceRequirement.PINNED_TEACHER_REVISION.value,
            EvidenceRequirement.TOKENIZER_FINGERPRINT_MATCH.value,
            EvidenceRequirement.FULL_LOGITS_AVAILABLE.value,
            EvidenceRequirement.LOCAL_WHITE_BOX.value,
            EvidenceRequirement.NETWORK_ISOLATION.value,
        ),
        "config_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["seed"],
            "properties": {"seed": {"type": "integer", "minimum": 0}},
        },
        "artifact_contract": ArtifactContract(
            required_outputs=("adapter", "tokenizer", "SHA256SUMS"),
        ),
        "metrics": ("primary_index",),
        "hardware": HardwareRequirements(
            min_gpu_memory_gib=16,
            approved_instance_types=("ml.g5.xlarge",),
            requires_network_isolation=True,
        ),
        "cost_model": CostModel(
            default_max_runtime_seconds=600,
            default_max_run_usd=10.0,
        ),
        "plugin_image": PluginImageBinding(
            image_uri=(f"123456789012.dkr.ecr.us-east-1.amazonaws.com/plugin@sha256:{'c' * 64}"),
            image_digest=f"sha256:{'c' * 64}",
        ),
        "reviewed_source": ReviewedSourceBinding(
            repository_uri="https://github.com/example/plugin",
            commit_sha="d" * 40,
            source_tree_sha256="e" * 64,
            review_record_sha256="f" * 64,
        ),
    }
    base.update(overrides)
    return base


def _request(technique_id: str, config: dict) -> TechniqueRequest:
    return TechniqueRequest(
        technique_id=technique_id,
        version="1.0.0",
        config=config,
    )


def test_unknown_capability_and_impossible_signal_rejected() -> None:
    with pytest.raises(TechniqueError) as unknown:
        TechniqueDescriptor.seal(
            **_external_kwargs(
                capabilities=(
                    TechniqueCapability.FULL_LOGITS.value,
                    TechniqueCapability.NETWORK_ISOLATED_PLUGIN.value,
                    "telepathy",
                )
            )
        )
    assert unknown.value.code is TechniqueErrorCode.TECHNIQUE_CAPABILITY_UNKNOWN
    with pytest.raises(TechniqueError) as impossible:
        TechniqueDescriptor.seal(
            **_external_kwargs(
                tokenizer_constraint=TokenizerConstraint.STUDENT_ONLY,
                capabilities=(TechniqueCapability.NETWORK_ISOLATED_PLUGIN.value,),
                evidence_requirements=(EvidenceRequirement.NETWORK_ISOLATION.value,),
            )
        )
    assert impossible.value.code is TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID


def test_invalid_schema_and_secret_fields_rejected(
    registry: TechniqueRegistry,
    logit_context: CompatibilityContext,
) -> None:
    with pytest.raises(TechniqueError) as invalid:
        TechniqueDescriptor.seal(
            **_external_kwargs(
                config_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"seed": {"type": 123}},
                }
            )
        )
    assert invalid.value.code is TechniqueErrorCode.TECHNIQUE_SCHEMA_MISMATCH
    with pytest.raises(TechniqueError, match="secret-like"):
        TechniqueDescriptor.seal(
            **_external_kwargs(
                config_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"api_key": {"type": "string"}},
                }
            )
        )
    descriptor = TechniqueDescriptor.seal(
        **_external_kwargs(
            config_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["options"],
                "properties": {
                    "options": {
                        "type": "object",
                        "additionalProperties": True,
                    }
                },
            }
        )
    )
    registry.register(descriptor)
    with pytest.raises(TechniqueError, match="secret-like"):
        registry.plan(
            _request(
                descriptor.technique_id,
                {"options": {"access_token": "must-not-enter-channel"}},
            ),
            logit_context,
        )


def test_schema_config_mismatch(
    registry: TechniqueRegistry,
    sequence_context: CompatibilityContext,
) -> None:
    with pytest.raises(TechniqueError) as excinfo:
        registry.plan(
            _request("sequence.v1", {"max_length": 512}),
            sequence_context,
        )
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH


def test_descriptor_is_frozen_and_hash_bound() -> None:
    descriptor = TechniqueDescriptor.seal(**_external_kwargs())
    with pytest.raises(ValidationError):
        descriptor.summary = "mutated"  # type: ignore[misc]
    with pytest.raises(TechniqueError) as tampered:
        TechniqueDescriptor.model_validate(
            {**descriptor.model_dump(mode="json"), "summary": "tampered"}
        )
    assert tampered.value.code is TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID


def test_digest_tag_and_field_confusion_rejected() -> None:
    with pytest.raises(TechniqueError) as tag:
        PluginImageBinding(
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/plugin:latest",
            image_digest=f"sha256:{'c' * 64}",
        )
    assert tag.value.code is TechniqueErrorCode.TECHNIQUE_DIGEST_INVALID
    with pytest.raises(TechniqueError) as mismatch:
        PluginImageBinding(
            image_uri=(f"123456789012.dkr.ecr.us-east-1.amazonaws.com/plugin@sha256:{'c' * 64}"),
            image_digest=f"sha256:{'a' * 64}",
        )
    assert mismatch.value.code is TechniqueErrorCode.TECHNIQUE_DIGEST_INVALID


def test_reviewed_source_requires_immutable_https_shape() -> None:
    with pytest.raises(ValidationError):
        ReviewedSourceBinding(
            repository_uri="https://user:password@example.com/repo?ref=main",
            commit_sha="d" * 40,
            source_tree_sha256="e" * 64,
            review_record_sha256="f" * 64,
        )


def test_divergent_collision_and_builtin_squatting(
    registry: TechniqueRegistry,
    external_descriptor: TechniqueDescriptor,
) -> None:
    first = registry.register(external_descriptor)
    assert registry.register(external_descriptor) is first
    payload = external_descriptor.canonical_payload()
    payload["summary"] = "divergent implementation"
    divergent = TechniqueDescriptor.seal(**payload)
    with pytest.raises(TechniqueError) as collision:
        registry.register(divergent)
    assert collision.value.code is TechniqueErrorCode.TECHNIQUE_VERSION_COLLISION
    with pytest.raises(TechniqueError) as squat:
        TechniqueDescriptor.seal(**_external_kwargs(technique_id="sequence.v1", version="99.0.0"))
    assert squat.value.code is TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID


def test_missing_instance_type_fails_closed() -> None:
    with pytest.raises(ValidationError):
        CompatibilityContext.model_validate(
            {
                "backend_kind": "local",
                "student_model_id": "student",
                "student_revision": "a" * 40,
                "tokenizer_sha256_student": "1" * 64,
                "chat_template_sha256_student": "2" * 64,
                "network_isolation": True,
            }
        )


@pytest.mark.parametrize(
    "field",
    [
        "tokenizer_sha256_teacher",
        "chat_template_sha256_teacher",
        "special_token_map_match",
        "full_logits_available",
        "local_white_box",
        "memory_dry_run_ok",
    ],
)
def test_required_logit_claim_absence_fails_closed(
    field: str,
    registry: TechniqueRegistry,
    logit_context: CompatibilityContext,
    logit_config: dict,
) -> None:
    bad = logit_context.model_copy(update={field: None})
    with pytest.raises(TechniqueError) as excinfo:
        registry.plan(_request("logit.v1", logit_config), bad)
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE


def test_config_identity_has_no_ambient_fallback(
    registry: TechniqueRegistry,
    sequence_context: CompatibilityContext,
    sequence_config: dict,
) -> None:
    missing = dict(sequence_config)
    missing.pop("student_model_id")
    with pytest.raises(TechniqueError) as absent:
        registry.plan(_request("sequence.v1", missing), sequence_context)
    assert absent.value.code is TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH
    mismatched = {**sequence_config, "student_model_id": "other/student"}
    with pytest.raises(TechniqueError) as mismatch:
        registry.plan(_request("sequence.v1", mismatched), sequence_context)
    assert mismatch.value.code is TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE


def test_every_context_identity_change_changes_protocol_hash(
    registry: TechniqueRegistry,
    sequence_context: CompatibilityContext,
    sequence_config: dict,
) -> None:
    base = registry.plan(_request("sequence.v1", sequence_config), sequence_context)
    variants: list[tuple[dict, CompatibilityContext]] = [
        (sequence_config, sequence_context.model_copy(update={"run_id": "other"})),
        (
            sequence_config,
            sequence_context.model_copy(update={"backend_kind": "sagemaker"}),
        ),
        (
            sequence_config,
            sequence_context.model_copy(update={"instance_type": "ml.g5.2xlarge"}),
        ),
    ]
    identities = {
        "student_model_id": ("student_model_id", "other/student"),
        "student_revision": ("student_revision", "9" * 40),
        "student_tokenizer_sha256": ("tokenizer_sha256_student", "8" * 64),
        "student_chat_template_sha256": (
            "chat_template_sha256_student",
            "7" * 64,
        ),
    }
    for config_field, (context_field, value) in identities.items():
        variants.append(
            (
                {**sequence_config, config_field: value},
                sequence_context.model_copy(update={context_field: value}),
            )
        )
    hashes = {
        registry.plan(_request("sequence.v1", config), context).protocol_sha256
        for config, context in variants
    }
    assert base.protocol_sha256 not in hashes
    assert len(hashes) == len(variants)


def test_teacher_identity_change_changes_protocol_hash(
    registry: TechniqueRegistry,
    logit_context: CompatibilityContext,
    logit_config: dict,
) -> None:
    base = registry.plan(_request("logit.v1", logit_config), logit_context)
    changed = registry.plan(
        _request(
            "logit.v1",
            {**logit_config, "teacher_revision": "8" * 40},
        ),
        logit_context.model_copy(update={"teacher_revision": "8" * 40}),
    )
    assert changed.protocol_sha256 != base.protocol_sha256


def test_recompute_tamper_and_config_order(
    registry: TechniqueRegistry,
    sequence_context: CompatibilityContext,
    sequence_config: dict,
) -> None:
    plan = registry.plan(_request("sequence.v1", sequence_config), sequence_context)
    assert recompute_protocol_hash(plan) == plan.protocol_sha256
    reordered = registry.plan(
        _request("sequence.v1", dict(reversed(sequence_config.items()))),
        sequence_context,
    )
    assert reordered.protocol_sha256 == plan.protocol_sha256
    payload = plan.model_dump(mode="json")
    payload["objective_fields"]["objective"] = "tampered"
    with pytest.raises(TechniqueError) as tampered:
        TechniquePlan.model_validate(payload)
    assert tampered.value.code is TechniqueErrorCode.TECHNIQUE_NONDETERMINISTIC


def test_plan_is_only_complete_preflight_and_lifecycle_is_retained(
    registry: TechniqueRegistry,
    sequence_context: CompatibilityContext,
    sequence_config: dict,
) -> None:
    assert not hasattr(registry, "validate")
    with pytest.raises(TechniqueError):
        registry.plan(
            _request("sequence.v1", sequence_config),
            sequence_context.model_copy(update={"usable_responses": None}),
        )
    plan = registry.plan(_request("sequence.v1", sequence_config), sequence_context)
    assert tuple(plan.lifecycle_history) == (
        "registered",
        "compatible",
        "planned",
    )
    assert registry.plan(_request("sequence.v1", sequence_config), sequence_context) is plan


def test_artifact_contract_and_exactly_one_execution(
    registry: TechniqueRegistry,
    sequence_context: CompatibilityContext,
    sequence_config: dict,
) -> None:
    with pytest.raises(TechniqueError) as contract:
        ArtifactContract(required_outputs=("adapter", "tokenizer"))
    assert contract.value.code is (TechniqueErrorCode.TECHNIQUE_ARTIFACT_CONTRACT_MISMATCH)
    plan = registry.plan(_request("sequence.v1", sequence_config), sequence_context)
    with pytest.raises(TechniqueError) as outputs:
        plan.validate_artifacts(
            {
                "adapter": "a" * 64,
                "tokenizer": "b" * 64,
                "chat_template": "c" * 64,
            }
        )
    assert outputs.value.code is (TechniqueErrorCode.TECHNIQUE_ARTIFACT_CONTRACT_MISMATCH)
    plan.validate_artifacts(
        {
            "adapter": "a" * 64,
            "tokenizer": "b" * 64,
            "chat_template": "c" * 64,
            "SHA256SUMS": "d" * 64,
        }
    )
    payload = plan.model_dump(mode="json")
    payload["training_load_plan"] = None
    with pytest.raises(TechniqueError, match="exactly one execution plan"):
        TechniquePlan.model_validate(payload)


def test_distillery_error_mapping_preserves_distinct_semantics() -> None:
    expected = {
        TechniqueErrorCode.TECHNIQUE_UNKNOWN: DistilleryErrorCode.TECHNIQUE_UNKNOWN,
        TechniqueErrorCode.TECHNIQUE_EXTERNAL_IMPORT_FORBIDDEN: (
            DistilleryErrorCode.TECHNIQUE_EXTERNAL_IMPORT_FORBIDDEN
        ),
        TechniqueErrorCode.TECHNIQUE_NONDETERMINISTIC: (
            DistilleryErrorCode.TECHNIQUE_NONDETERMINISTIC
        ),
    }
    for technique_code, distillery_code in expected.items():
        assert technique_error(technique_code, "x").as_distillery_error().code is (distillery_code)


def test_descriptor_roundtrip_from_path(
    tmp_path: Path,
    external_descriptor: TechniqueDescriptor,
) -> None:
    assert external_descriptor.schema_version == DESCRIPTOR_SCHEMA_VERSION
    path = tmp_path / "technique.json"
    path.write_text(
        json.dumps(external_descriptor.model_dump(mode="json")),
        encoding="utf-8",
    )
    loaded = TechniqueRegistry.with_builtins().register_from_path(path)
    assert loaded.descriptor_sha256 == external_descriptor.descriptor_sha256
