"""Adversarial BYODT tests: fail closed, no silent fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from distillery.techniques import (
    ArtifactContract,
    CostModel,
    EvidenceRequirement,
    ExecutionKind,
    HardwareRequirements,
    PluginImageBinding,
    ReviewedSourceBinding,
    TeacherSignal,
    TechniqueCapability,
    TechniqueDescriptor,
    TechniqueError,
    TechniqueErrorCode,
    TechniqueRegistry,
    TechniqueRequest,
    TokenizerConstraint,
    forbid_control_plane_import,
)
from distillery.techniques.channel import write_channel_plan
from distillery.techniques.descriptor import DESCRIPTOR_SCHEMA_VERSION


def _base_external_kwargs(**overrides):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["seed"],
        "properties": {"seed": {"type": "integer", "minimum": 0}},
    }
    base = {
        "technique_id": "hackathon.dhilan.reverse_kl",
        "version": "1.0.0",
        "display_name": "reverse KL",
        "summary": "external technique",
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
            EvidenceRequirement.PLUGIN_IMAGE_DIGEST.value,
            EvidenceRequirement.REVIEWED_SOURCE_BINDING.value,
            EvidenceRequirement.NETWORK_ISOLATION.value,
        ),
        "config_schema": schema,
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


def test_unknown_capability_rejected() -> None:
    with pytest.raises(TechniqueError) as excinfo:
        TechniqueDescriptor.seal(
            **_base_external_kwargs(
                capabilities=(
                    TechniqueCapability.FULL_LOGITS.value,
                    "telepathy",
                    TechniqueCapability.NETWORK_ISOLATED_PLUGIN.value,
                )
            )
        )
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_CAPABILITY_UNKNOWN


def test_schema_config_mismatch(registry, sequence_context) -> None:
    with pytest.raises(TechniqueError) as excinfo:
        registry.plan(
            TechniqueRequest(
                technique_id="sequence.v1",
                version="1.0.0",
                config={"max_length": 512},  # missing required fields
            ),
            sequence_context,
        )
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH


def test_mutable_descriptor_hash_tamper_detected() -> None:
    descriptor = TechniqueDescriptor.seal(**_base_external_kwargs())
    with pytest.raises(TechniqueError) as excinfo:
        TechniqueDescriptor.model_validate(
            {
                **descriptor.model_dump(mode="json"),
                "summary": "tampered summary",
                # keep old hash → integrity failure
            }
        )
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID


def test_frozen_descriptor_rejects_assignment() -> None:
    descriptor = TechniqueDescriptor.seal(**_base_external_kwargs())
    with pytest.raises(ValidationError):
        descriptor.summary = "mutated"  # type: ignore[misc]


def test_digest_tag_confusion_rejected() -> None:
    with pytest.raises(TechniqueError) as excinfo:
        PluginImageBinding(
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/plugin:latest",
            image_digest=f"sha256:{'c' * 64}",
        )
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_DIGEST_INVALID

    with pytest.raises(TechniqueError):
        TechniqueDescriptor.seal(
            **_base_external_kwargs(
                plugin_image={
                    "image_uri": ("123456789012.dkr.ecr.us-east-1.amazonaws.com/plugin:latest"),
                    "image_digest": f"sha256:{'c' * 64}",
                }
            )
        )


def test_digest_field_uri_mismatch() -> None:
    with pytest.raises(TechniqueError) as excinfo:
        PluginImageBinding(
            image_uri=(f"123456789012.dkr.ecr.us-east-1.amazonaws.com/plugin@sha256:{'c' * 64}"),
            image_digest=f"sha256:{'a' * 64}",
        )
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_DIGEST_INVALID


def test_version_collision(registry, external_descriptor) -> None:
    registry.register(external_descriptor)
    with pytest.raises(TechniqueError) as excinfo:
        registry.register(external_descriptor)
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_VERSION_COLLISION


def test_incompatible_tokenizer_for_logit(registry, logit_context) -> None:
    bad = logit_context.model_copy(update={"tokenizer_sha256_teacher": "9" * 64})
    with pytest.raises(TechniqueError) as excinfo:
        registry.plan(
            TechniqueRequest(
                technique_id="logit.v1",
                version="1.0.0",
                config={
                    "max_length": 512,
                    "max_completion": 160,
                    "seed": 17,
                    "temperature": 2.0,
                    "kd_weight": 0.7,
                    "hard_ce_weight": 0.3,
                },
            ),
            bad,
        )
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE
    assert "tokenizer" in str(excinfo.value.payload.details).lower() or any(
        "tokenizer" in reason
        for reason in excinfo.value.payload.details.get("rejected_reasons", [])
    )


def test_incompatible_logit_access(registry, logit_context) -> None:
    bad = logit_context.model_copy(update={"local_white_box": False})
    with pytest.raises(TechniqueError) as excinfo:
        registry.plan(
            TechniqueRequest(
                technique_id="logit.v1",
                version="1.0.0",
                config={
                    "max_length": 512,
                    "max_completion": 160,
                    "seed": 17,
                    "temperature": 2.0,
                    "kd_weight": 0.7,
                    "hard_ce_weight": 0.3,
                },
            ),
            bad,
        )
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE


def test_artifact_contract_mismatch_on_seal() -> None:
    with pytest.raises(ValidationError):
        ArtifactContract(required_outputs=())


def test_external_without_network_isolation_rejected(
    registry, logit_context, external_descriptor
) -> None:
    registry.register(external_descriptor)
    bad = logit_context.model_copy(update={"network_isolation": False})
    with pytest.raises(TechniqueError) as excinfo:
        registry.plan(
            TechniqueRequest(
                technique_id="hackathon.dhilan.reverse_kl",
                version="1.0.0",
                config={
                    "max_length": 512,
                    "max_completion": 160,
                    "seed": 17,
                    "temperature": 2.0,
                },
            ),
            bad,
        )
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE


def test_protocol_hash_deterministic(registry, sequence_context) -> None:
    request = TechniqueRequest(
        technique_id="sequence.v1",
        version="1.0.0",
        config={"max_length": 512, "max_completion": 160, "seed": 17},
    )
    first = registry.plan(request, sequence_context)
    second = registry.plan(request, sequence_context)
    assert first.protocol_sha256 == second.protocol_sha256
    assert first.plan_hash() == second.plan_hash()


def test_config_key_order_does_not_change_hash(registry, sequence_context) -> None:
    a = registry.plan(
        TechniqueRequest(
            technique_id="sequence.v1",
            version="1.0.0",
            config={"seed": 17, "max_completion": 160, "max_length": 512},
        ),
        sequence_context,
    )
    b = registry.plan(
        TechniqueRequest(
            technique_id="sequence.v1",
            version="1.0.0",
            config={"max_length": 512, "max_completion": 160, "seed": 17},
        ),
        sequence_context,
    )
    assert a.config_sha256 == b.config_sha256
    assert a.protocol_sha256 == b.protocol_sha256


def test_external_import_forbidden(external_descriptor) -> None:
    from distillery.techniques.adapters.external import ExternalContainerAdapter

    adapter = ExternalContainerAdapter(external_descriptor)
    with pytest.raises(TechniqueError) as excinfo:
        adapter.import_plugin("hackathon.dhilan.reverse_kl.train")
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_EXTERNAL_IMPORT_FORBIDDEN
    with pytest.raises(TechniqueError):
        forbid_control_plane_import("anything.plugin")


def test_channel_rejects_extra_json(
    tmp_path: Path, registry, logit_context, external_descriptor
) -> None:
    registry.register(external_descriptor)
    plan = registry.plan(
        TechniqueRequest(
            technique_id="hackathon.dhilan.reverse_kl",
            version="1.0.0",
            config={
                "max_length": 512,
                "max_completion": 160,
                "seed": 17,
                "temperature": 2.0,
            },
        ),
        logit_context,
    )
    channel_dir = tmp_path / "channel"
    from distillery.techniques.channel import TechniqueChannelContract

    contract = TechniqueChannelContract.model_validate(dict(plan.channel_contract))
    write_channel_plan(
        channel_dir,
        contract=contract,
        plan_payload=plan.model_dump(mode="json"),
    )
    (channel_dir / "extra.json").write_text("{}", encoding="utf-8")
    from distillery.techniques.channel import load_channel_plan

    with pytest.raises(TechniqueError) as excinfo:
        load_channel_plan(channel_dir)
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID


def test_descriptor_schema_version_locked() -> None:
    descriptor = TechniqueDescriptor.seal(**_base_external_kwargs())
    assert descriptor.schema_version == DESCRIPTOR_SCHEMA_VERSION


def test_register_from_path_roundtrip(
    tmp_path: Path, external_descriptor: TechniqueDescriptor
) -> None:
    path = tmp_path / "tech.json"
    path.write_text(json.dumps(external_descriptor.model_dump(mode="json")), encoding="utf-8")
    registry = TechniqueRegistry.with_builtins()
    loaded = registry.register_from_path(path)
    assert loaded.descriptor_sha256 == external_descriptor.descriptor_sha256
