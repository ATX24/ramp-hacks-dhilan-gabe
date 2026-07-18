#!/usr/bin/env python3
"""Seal the example reverse-KL BYODT descriptor (plan-only SDK helper)."""

from __future__ import annotations

import json
from pathlib import Path

from distillery.techniques.capabilities import (
    EvidenceRequirement,
    TechniqueCapability,
)
from distillery.techniques.descriptor import (
    ArtifactContract,
    CostModel,
    ExecutionKind,
    HardwareRequirements,
    PluginImageBinding,
    ReviewedSourceBinding,
    TeacherSignal,
    TechniqueDescriptor,
    TokenizerConstraint,
)

HERE = Path(__file__).resolve().parent
DIGEST = "c" * 64
IMAGE_URI = (
    f"123456789012.dkr.ecr.us-east-1.amazonaws.com/distillery-technique-reverse-kl@sha256:{DIGEST}"
)


def build() -> TechniqueDescriptor:
    schema = json.loads((HERE / "config.schema.json").read_text(encoding="utf-8"))
    return TechniqueDescriptor.seal(
        technique_id="hackathon.dhilan.reverse_kl",
        version="1.0.0",
        display_name="Hackathon reverse KL v1",
        summary=(
            "Example Bring-Your-Own technique: a plan-only reverse KL contract "
            "for future network-isolated backend integration."
        ),
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
            EvidenceRequirement.FULL_LOGITS_AVAILABLE.value,
            EvidenceRequirement.LOCAL_WHITE_BOX.value,
            EvidenceRequirement.NETWORK_ISOLATION.value,
        ),
        config_schema=schema,
        artifact_contract=ArtifactContract(
            required_outputs=("adapter", "tokenizer", "chat_template", "SHA256SUMS"),
            optional_outputs=("merged", "technique_metrics"),
        ),
        metrics=("primary_index", "reverse_kl_loss", "json_schema_validity"),
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
            image_uri=IMAGE_URI,
            image_digest=f"sha256:{DIGEST}",
        ),
        reviewed_source=ReviewedSourceBinding(
            repository_uri="https://github.com/example/distillery-reverse-kl",
            commit_sha="d" * 40,
            source_tree_sha256="e" * 64,
            review_record_sha256="f" * 64,
        ),
    )


def main() -> None:
    descriptor = build()
    out = HERE / "technique.json"
    out.write_text(
        json.dumps(descriptor.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(out)
    print(descriptor.descriptor_sha256)


if __name__ == "__main__":
    main()
