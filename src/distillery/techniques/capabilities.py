"""Closed capability vocabulary for technique descriptors."""

from __future__ import annotations

from enum import StrEnum

from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error


class TechniqueCapability(StrEnum):
    """Capabilities a technique may declare. Unknown strings are rejected."""

    HARD_TARGET_SEQUENCE = "hard_target_sequence"
    FULL_LOGITS = "full_logits"
    LOCAL_WHITE_BOX = "local_white_box"
    BLACK_BOX_RESPONSE = "black_box_response"
    QLORA_ADAPTATION = "qlora_adaptation"
    NETWORK_ISOLATED_PLUGIN = "network_isolated_plugin"
    DETERMINISTIC_PLAN = "deterministic_plan"
    COMPLETION_ONLY_CE = "completion_only_ce"
    FORWARD_KL_PLUS_HARD_CE = "forward_kl_plus_hard_ce"
    CUSTOM_OBJECTIVE = "custom_objective"


class EvidenceRequirement(StrEnum):
    """Evidence keys a technique may require before planning."""

    PINNED_STUDENT_REVISION = "pinned_student_revision"
    PINNED_TEACHER_REVISION = "pinned_teacher_revision"
    TOKENIZER_FINGERPRINT_MATCH = "tokenizer_fingerprint_match"
    SPECIAL_TOKEN_MAP_MATCH = "special_token_map_match"
    CHAT_TEMPLATE_COMPATIBLE = "chat_template_compatible"
    MEMORY_DRY_RUN_OK = "memory_dry_run_ok"
    LOCAL_WHITE_BOX = "local_white_box"
    USABLE_RESPONSES = "usable_responses"
    PLUGIN_IMAGE_DIGEST = "plugin_image_digest"
    REVIEWED_SOURCE_BINDING = "reviewed_source_binding"
    NETWORK_ISOLATION = "network_isolation"


KNOWN_CAPABILITIES: frozenset[str] = frozenset(member.value for member in TechniqueCapability)
KNOWN_EVIDENCE: frozenset[str] = frozenset(member.value for member in EvidenceRequirement)


def require_known_capabilities(capabilities: tuple[str, ...]) -> tuple[str, ...]:
    unknown = sorted(set(capabilities) - KNOWN_CAPABILITIES)
    if unknown:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CAPABILITY_UNKNOWN,
            "technique declares unknown capabilities",
            details={"unknown_capabilities": unknown},
        )
    return capabilities


def require_known_evidence(requirements: tuple[str, ...]) -> tuple[str, ...]:
    unknown = sorted(set(requirements) - KNOWN_EVIDENCE)
    if unknown:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CAPABILITY_UNKNOWN,
            "technique declares unknown evidence requirements",
            details={"unknown_evidence_requirements": unknown},
        )
    return requirements


__all__ = [
    "KNOWN_CAPABILITIES",
    "KNOWN_EVIDENCE",
    "EvidenceRequirement",
    "TechniqueCapability",
    "require_known_capabilities",
    "require_known_evidence",
]
