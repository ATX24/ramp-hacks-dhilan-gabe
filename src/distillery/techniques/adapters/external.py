"""External technique adapter: plan-only, container channel execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from distillery.techniques.channel import (
    build_channel_contract,
    forbid_control_plane_import,
)
from distillery.techniques.compatibility import (
    CompatibilityContext,
    negotiate_compatibility,
)
from distillery.techniques.descriptor import ExecutionKind, TechniqueDescriptor
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error
from distillery.techniques.lifecycle import TechniqueLifecycle
from distillery.techniques.protocol import assert_protocol_deterministic
from distillery.techniques.runtime import (
    ExternalExecutionPlan,
    LossContract,
    TechniquePlan,
)
from distillery.techniques.schema import validate_config_against_schema


class ExternalContainerAdapter:
    """
    Adapter for Bring-Your-Own techniques.

    Validates and plans only. Runtime execution is delegated to the digest-
    pinned, network-isolated container referenced by the sealed descriptor.
    Plugin Python is never imported into the control plane.
    """

    def __init__(self, descriptor: TechniqueDescriptor) -> None:
        if descriptor.execution is not ExecutionKind.EXTERNAL_CONTAINER:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "ExternalContainerAdapter requires external_container execution",
                details={"execution": descriptor.execution.value},
            )
        self.descriptor = descriptor

    def validate(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> str:
        return validate_config_against_schema(
            config,
            self.descriptor.config_schema,
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
        )

    def plan(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> TechniquePlan:
        config_sha256 = self.validate(config, context)
        compatibility = negotiate_compatibility(self.descriptor, context)
        # Hash the channel identity without protocol_sha256 to avoid a cycle,
        # then bind the resulting protocol hash into the sealed channel.
        channel_identity = build_channel_contract(
            descriptor=self.descriptor,
            config_sha256=config_sha256,
            protocol_sha256="0" * 64,
        ).model_dump(mode="json")
        channel_identity.pop("protocol_sha256", None)
        protocol_sha256 = assert_protocol_deterministic(
            descriptor=self.descriptor,
            config_sha256=config_sha256,
            compatibility=compatibility,
            channel_contract=channel_identity,
        )
        channel = build_channel_contract(
            descriptor=self.descriptor,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )
        assert self.descriptor.plugin_image is not None
        assert self.descriptor.reviewed_source is not None
        objective = {
            "recipe_id": self.descriptor.technique_id,
            "mode": "external_container",
            "objective": "custom",
            "signal": self.descriptor.teacher_signal.value,
            "execution": ExecutionKind.EXTERNAL_CONTAINER.value,
        }
        loss = LossContract(
            objective="custom",
            signal=self.descriptor.teacher_signal.value,
            mode="external_container",
            fields=objective,
        )
        external = ExternalExecutionPlan(
            image_uri=self.descriptor.plugin_image.image_uri,
            image_digest=self.descriptor.plugin_image.image_digest,
            reviewed_source_commit=self.descriptor.reviewed_source.commit_sha,
            reviewed_source_tree_sha256=(self.descriptor.reviewed_source.source_tree_sha256),
        )
        return TechniquePlan(
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
            descriptor_sha256=self.descriptor.descriptor_sha256,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
            lifecycle=TechniqueLifecycle.PLANNED,
            compatibility=compatibility,
            loss=loss,
            training_load_plan=None,
            external_execution=external,
            objective_fields=objective,
            channel_contract=channel.model_dump(mode="json"),
        )

    def import_plugin(self, module_name: str) -> None:
        """Explicitly forbidden control-plane import path."""
        forbid_control_plane_import(module_name)


__all__ = ["ExternalContainerAdapter"]
