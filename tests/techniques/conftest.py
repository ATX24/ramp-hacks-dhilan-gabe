"""Shared fixtures for BYODT technique tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from distillery.techniques import (
    ArtifactContract,
    CompatibilityContext,
    CostModel,
    EvidenceRequirement,
    ExecutionKind,
    HardwareRequirements,
    PluginImageBinding,
    ReviewedSourceBinding,
    TeacherSignal,
    TechniqueCapability,
    TechniqueDescriptor,
    TechniqueRegistry,
    TokenizerConstraint,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLE_DIR = ROOT / "examples" / "byodt" / "reverse_kl_v1"
REVISION_A = "a" * 40
REVISION_B = "b" * 40
DIGEST = "1" * 64
DIGEST_T = "2" * 64
PLUGIN_DIGEST = "c" * 64
PLUGIN_URI = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/distillery-technique-reverse-kl"
    f"@sha256:{PLUGIN_DIGEST}"
)


@pytest.fixture
def registry() -> TechniqueRegistry:
    return TechniqueRegistry.with_builtins()


@pytest.fixture
def sequence_context() -> CompatibilityContext:
    return CompatibilityContext(
        backend_kind="local",
        student_model_id="Qwen/Qwen2.5-0.5B",
        student_revision=REVISION_A,
        usable_responses=True,
        network_isolation=True,
        instance_type="ml.g5.xlarge",
    )


@pytest.fixture
def logit_context() -> CompatibilityContext:
    return CompatibilityContext(
        backend_kind="sagemaker",
        student_model_id="Qwen/Qwen2.5-0.5B",
        student_revision=REVISION_A,
        teacher_model_id="Qwen/Qwen2.5-1.5B",
        teacher_revision=REVISION_B,
        tokenizer_sha256_student=DIGEST,
        tokenizer_sha256_teacher=DIGEST,
        chat_template_sha256_student=DIGEST_T,
        chat_template_sha256_teacher=DIGEST_T,
        special_token_map_match=True,
        local_white_box=True,
        memory_dry_run_ok=True,
        usable_responses=True,
        network_isolation=True,
        instance_type="ml.g5.xlarge",
    )


@pytest.fixture
def external_descriptor() -> TechniqueDescriptor:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["max_length", "max_completion", "seed", "temperature"],
        "properties": {
            "max_length": {"type": "integer", "minimum": 2},
            "max_completion": {"type": "integer", "minimum": 1},
            "seed": {"type": "integer", "minimum": 0},
            "temperature": {"type": "number", "exclusiveMinimum": 0},
        },
    }
    return TechniqueDescriptor.seal(
        technique_id="hackathon.dhilan.reverse_kl",
        version="1.0.0",
        display_name="Hackathon reverse KL v1",
        summary="Example external reverse-KL technique",
        execution=ExecutionKind.EXTERNAL_CONTAINER,
        teacher_signal=TeacherSignal.FULL_LOGITS,
        tokenizer_constraint=TokenizerConstraint.EXACT_MATCH,
        capabilities=(
            TechniqueCapability.FULL_LOGITS.value,
            TechniqueCapability.LOCAL_WHITE_BOX.value,
            TechniqueCapability.NETWORK_ISOLATED_PLUGIN.value,
            TechniqueCapability.CUSTOM_OBJECTIVE.value,
            TechniqueCapability.DETERMINISTIC_PLAN.value,
        ),
        evidence_requirements=(
            EvidenceRequirement.PINNED_STUDENT_REVISION.value,
            EvidenceRequirement.PINNED_TEACHER_REVISION.value,
            EvidenceRequirement.TOKENIZER_FINGERPRINT_MATCH.value,
            EvidenceRequirement.SPECIAL_TOKEN_MAP_MATCH.value,
            EvidenceRequirement.CHAT_TEMPLATE_COMPATIBLE.value,
            EvidenceRequirement.LOCAL_WHITE_BOX.value,
            EvidenceRequirement.PLUGIN_IMAGE_DIGEST.value,
            EvidenceRequirement.REVIEWED_SOURCE_BINDING.value,
            EvidenceRequirement.NETWORK_ISOLATION.value,
        ),
        config_schema=schema,
        artifact_contract=ArtifactContract(
            required_outputs=("adapter", "tokenizer", "chat_template", "SHA256SUMS"),
        ),
        metrics=("primary_index", "reverse_kl_loss"),
        hardware=HardwareRequirements(
            min_gpu_memory_gib=16,
            approved_instance_types=("ml.g5.xlarge", "ml.g5.2xlarge"),
            requires_network_isolation=True,
        ),
        cost_model=CostModel(
            default_max_runtime_seconds=2700,
            default_max_run_usd=30.0,
        ),
        plugin_image=PluginImageBinding(
            image_uri=PLUGIN_URI,
            image_digest=f"sha256:{PLUGIN_DIGEST}",
        ),
        reviewed_source=ReviewedSourceBinding(
            repository_uri="https://github.com/example/distillery-reverse-kl",
            commit_sha="d" * 40,
            source_tree_sha256="e" * 64,
            review_record_sha256="f" * 64,
        ),
    )


@pytest.fixture
def example_technique_json(tmp_path: Path, external_descriptor: TechniqueDescriptor) -> Path:
    path = tmp_path / "technique.json"
    path.write_text(
        json.dumps(external_descriptor.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path
