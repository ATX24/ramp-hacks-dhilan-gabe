"""Sealed built-in technique descriptors for sequence.v1 and logit.v1."""

from __future__ import annotations

from distillery.techniques.capabilities import EvidenceRequirement, TechniqueCapability
from distillery.techniques.descriptor import (
    ArtifactContract,
    CostModel,
    ExecutionKind,
    HardwareRequirements,
    TeacherSignal,
    TechniqueDescriptor,
    TokenizerConstraint,
)

_DEFAULT_ARTIFACT = ArtifactContract(
    required_outputs=("adapter", "tokenizer", "chat_template", "SHA256SUMS"),
    optional_outputs=("merged",),
)

_DEFAULT_HARDWARE = HardwareRequirements(
    min_gpu_memory_gib=16,
    approved_instance_types=("ml.g5.2xlarge", "ml.g5.xlarge"),
    requires_network_isolation=True,
)

_DEFAULT_COST = CostModel(
    default_max_runtime_seconds=45 * 60,
    default_max_run_usd=25.0,
)

SEQUENCE_CONFIG_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "max_length",
        "max_completion",
        "seed",
        "student_model_id",
        "student_revision",
        "student_tokenizer_sha256",
        "student_chat_template_sha256",
        "require_nonempty_response",
        "require_json_object_response",
        "pad_token_id",
    ],
    "properties": {
        "max_length": {"type": "integer", "minimum": 2},
        "max_completion": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer", "minimum": 0},
        "student_model_id": {"type": "string", "minLength": 1},
        "student_revision": {
            "type": "string",
            "pattern": "^[0-9a-f]{40}$",
        },
        "student_tokenizer_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "student_chat_template_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "require_nonempty_response": {"type": "boolean"},
        "require_json_object_response": {"type": "boolean"},
        "pad_token_id": {"type": ["integer", "null"], "minimum": 0},
    },
}

LOGIT_CONFIG_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "max_length",
        "max_completion",
        "seed",
        "temperature",
        "kd_weight",
        "hard_ce_weight",
        "vocab_chunk_size",
        "student_model_id",
        "student_revision",
        "student_tokenizer_sha256",
        "student_chat_template_sha256",
        "teacher_model_id",
        "teacher_revision",
        "teacher_tokenizer_sha256",
        "teacher_chat_template_sha256",
    ],
    "properties": {
        "max_length": {"type": "integer", "minimum": 2},
        "max_completion": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer", "minimum": 0},
        "temperature": {"type": "number", "exclusiveMinimum": 0},
        "kd_weight": {"type": "number", "minimum": 0, "maximum": 1},
        "hard_ce_weight": {"type": "number", "minimum": 0, "maximum": 1},
        "vocab_chunk_size": {"type": "integer", "minimum": 1},
        "student_model_id": {"type": "string", "minLength": 1},
        "student_revision": {
            "type": "string",
            "pattern": "^[0-9a-f]{40}$",
        },
        "student_tokenizer_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "student_chat_template_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "teacher_model_id": {"type": "string", "minLength": 1},
        "teacher_revision": {
            "type": "string",
            "pattern": "^[0-9a-f]{40}$",
        },
        "teacher_tokenizer_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "teacher_chat_template_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
    },
}


def sequence_v1_descriptor() -> TechniqueDescriptor:
    return TechniqueDescriptor.seal(
        technique_id="sequence.v1",
        version="1.0.0",
        display_name="Sequence distillation v1",
        summary=(
            "Completion-only SFT/QLoRA on imported or teacher-generated responses; "
            "student retokenizes text."
        ),
        execution=ExecutionKind.BUILTIN,
        teacher_signal=TeacherSignal.HARD_TARGET_SEQUENCE,
        tokenizer_constraint=TokenizerConstraint.STUDENT_ONLY,
        capabilities=(
            TechniqueCapability.HARD_TARGET_SEQUENCE.value,
            TechniqueCapability.BLACK_BOX_RESPONSE.value,
            TechniqueCapability.QLORA_ADAPTATION.value,
            TechniqueCapability.COMPLETION_ONLY_CE.value,
            TechniqueCapability.DETERMINISTIC_PLAN.value,
        ),
        evidence_requirements=(
            EvidenceRequirement.PINNED_STUDENT_REVISION.value,
            EvidenceRequirement.USABLE_RESPONSES.value,
        ),
        config_schema=SEQUENCE_CONFIG_SCHEMA,
        artifact_contract=_DEFAULT_ARTIFACT,
        metrics=("primary_index", "json_schema_validity", "completion_tokens"),
        hardware=_DEFAULT_HARDWARE,
        cost_model=_DEFAULT_COST,
    )


def logit_v1_descriptor() -> TechniqueDescriptor:
    return TechniqueDescriptor.seal(
        technique_id="logit.v1",
        version="1.0.0",
        display_name="Logit distillation v1",
        summary=(
            "Local white-box forward KL on teacher-forced completion positions "
            "with hard-target CE mixing."
        ),
        execution=ExecutionKind.BUILTIN,
        teacher_signal=TeacherSignal.FULL_LOGITS,
        tokenizer_constraint=TokenizerConstraint.EXACT_MATCH,
        capabilities=(
            TechniqueCapability.FULL_LOGITS.value,
            TechniqueCapability.LOCAL_WHITE_BOX.value,
            TechniqueCapability.QLORA_ADAPTATION.value,
            TechniqueCapability.FORWARD_KL_PLUS_HARD_CE.value,
            TechniqueCapability.DETERMINISTIC_PLAN.value,
        ),
        evidence_requirements=(
            EvidenceRequirement.PINNED_STUDENT_REVISION.value,
            EvidenceRequirement.PINNED_TEACHER_REVISION.value,
            EvidenceRequirement.TOKENIZER_FINGERPRINT_MATCH.value,
            EvidenceRequirement.SPECIAL_TOKEN_MAP_MATCH.value,
            EvidenceRequirement.CHAT_TEMPLATE_COMPATIBLE.value,
            EvidenceRequirement.FULL_LOGITS_AVAILABLE.value,
            EvidenceRequirement.LOCAL_WHITE_BOX.value,
            EvidenceRequirement.MEMORY_DRY_RUN_OK.value,
        ),
        config_schema=LOGIT_CONFIG_SCHEMA,
        artifact_contract=_DEFAULT_ARTIFACT,
        metrics=(
            "primary_index",
            "json_schema_validity",
            "kd_loss",
            "hard_ce_loss",
        ),
        hardware=_DEFAULT_HARDWARE,
        cost_model=_DEFAULT_COST,
    )


def builtin_descriptors() -> tuple[TechniqueDescriptor, ...]:
    return (sequence_v1_descriptor(), logit_v1_descriptor())


__all__ = [
    "LOGIT_CONFIG_SCHEMA",
    "SEQUENCE_CONFIG_SCHEMA",
    "builtin_descriptors",
    "logit_v1_descriptor",
    "sequence_v1_descriptor",
]
