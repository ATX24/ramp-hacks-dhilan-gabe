"""External technique adapter: plan-only identity for future backend wiring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from distillery.techniques.compatibility import (
    CompatibilityContext,
    negotiate_compatibility,
)
from distillery.techniques.descriptor import ExecutionKind, TechniqueDescriptor
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error
from distillery.techniques.lifecycle import TechniqueLifecycle
from distillery.techniques.runtime import (
    ExternalExecutionPlan,
    LossContract,
    ResolvedHardwarePlan,
    TechniquePlan,
)
from distillery.techniques.schema import validate_config_against_schema


class ExternalContainerAdapter:
    """
    Produce a sealed plan for future trainer/backend integration.

    This adapter does not execute, import plugin code, or enforce isolation.
    """

    def __init__(self, descriptor: TechniqueDescriptor) -> None:
        if descriptor.execution is not ExecutionKind.EXTERNAL_CONTAINER:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "ExternalContainerAdapter requires external_container execution",
            )
        self.descriptor = descriptor

    def plan(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> TechniquePlan:
        resolved, config_sha256 = validate_config_against_schema(
            config,
            self.descriptor.config_schema,
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
        )
        compatibility = negotiate_compatibility(self.descriptor, context)
        if self.descriptor.plugin_image is None or self.descriptor.reviewed_source is None:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "external descriptor lacks sealed image/source identity",
            )
        objective = {
            "recipe_id": self.descriptor.technique_id,
            "mode": "external_container",
            "objective": "custom",
            "signal": self.descriptor.teacher_signal.value,
            "execution": ExecutionKind.EXTERNAL_CONTAINER.value,
        }
        source = self.descriptor.reviewed_source
        image = self.descriptor.plugin_image
        return TechniquePlan.seal(
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
            descriptor=self.descriptor,
            descriptor_sha256=self.descriptor.descriptor_sha256,
            resolved_config=resolved,
            config_sha256=config_sha256,
            environment=context,
            lifecycle=TechniqueLifecycle.PLANNED,
            lifecycle_history=(
                TechniqueLifecycle.REGISTERED,
                TechniqueLifecycle.COMPATIBLE,
                TechniqueLifecycle.PLANNED,
            ),
            compatibility=compatibility,
            hardware=ResolvedHardwarePlan(
                backend_kind=context.backend_kind,
                instance_type=context.instance_type,
                approved_instance_types=self.descriptor.hardware.approved_instance_types,
                min_gpu_memory_gib=self.descriptor.hardware.min_gpu_memory_gib,
                network_isolation_required=(self.descriptor.hardware.requires_network_isolation),
                network_isolation_claimed=context.network_isolation,
            ),
            artifact_contract=self.descriptor.artifact_contract,
            loss=LossContract(
                objective="custom",
                signal=self.descriptor.teacher_signal.value,
                mode="external_container",
                fields=objective,
            ),
            adapter_config=resolved,
            training_load_plan=None,
            external_execution=ExternalExecutionPlan(
                image_uri=image.image_uri,
                image_digest=image.image_digest,
                reviewed_source_repository=source.repository_uri,
                reviewed_source_commit=source.commit_sha,
                reviewed_source_tree_sha256=source.source_tree_sha256,
                reviewed_source_review_record_sha256=source.review_record_sha256,
            ),
            objective_fields=objective,
        )


__all__ = ["ExternalContainerAdapter"]
