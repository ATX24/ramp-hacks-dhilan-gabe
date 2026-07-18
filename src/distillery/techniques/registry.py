"""Technique registry: resolve built-ins and external descriptors without fallback."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field, StrictStr

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.techniques.adapters.builtin import (
    BuiltinLogitAdapter,
    BuiltinSequenceAdapter,
)
from distillery.techniques.adapters.external import ExternalContainerAdapter
from distillery.techniques.builtins import builtin_descriptors
from distillery.techniques.compatibility import CompatibilityContext
from distillery.techniques.descriptor import ExecutionKind, TechniqueDescriptor
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error
from distillery.techniques.lifecycle import TechniqueLifecycle, advance_lifecycle
from distillery.techniques.runtime import RuntimeAdapter, TechniquePlan


class TechniqueRequest(FrozenModel):
    """Caller request at the technique seam. Version is required (no latest)."""

    technique_id: StrictStr = Field(min_length=1)
    version: StrictStr = Field(min_length=1)
    config: FrozenJsonObject = Field(default_factory=dict)


class TechniqueRegistry:
    """
    Deep-module registry at the BYODT seam.

    Callers resolve by exact ``technique_id@version``. Missing ids/versions and
    unknown capabilities fail loud. There is no silent fallback to sequence.v1.
    """

    def __init__(self) -> None:
        self._descriptors: dict[str, TechniqueDescriptor] = {}
        self._adapters: dict[str, RuntimeAdapter] = {}
        self._lifecycle: dict[str, TechniqueLifecycle] = {}

    @classmethod
    def with_builtins(cls) -> TechniqueRegistry:
        registry = cls()
        for descriptor in builtin_descriptors():
            registry.register(descriptor, lifecycle=TechniqueLifecycle.REGISTERED)
        return registry

    def register(
        self,
        descriptor: TechniqueDescriptor,
        *,
        lifecycle: TechniqueLifecycle = TechniqueLifecycle.REGISTERED,
    ) -> TechniqueDescriptor:
        descriptor.assert_integrity()
        key = descriptor.technique_key
        if key in self._descriptors:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_VERSION_COLLISION,
                "technique id/version already registered",
                details={
                    "technique_id": descriptor.technique_id,
                    "version": descriptor.version,
                    "existing_descriptor_sha256": self._descriptors[key].descriptor_sha256,
                    "incoming_descriptor_sha256": descriptor.descriptor_sha256,
                },
            )
        if lifecycle not in {
            TechniqueLifecycle.VALIDATED,
            TechniqueLifecycle.REGISTERED,
        }:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_LIFECYCLE_INVALID,
                "register requires validated or registered lifecycle",
                details={"lifecycle": lifecycle.value},
            )
        adapter = _adapter_for(descriptor)
        self._descriptors[key] = descriptor
        self._adapters[key] = adapter
        self._lifecycle[key] = lifecycle
        return descriptor

    def register_from_path(self, path: Path) -> TechniqueDescriptor:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "descriptor_sha256" in payload:
            descriptor = TechniqueDescriptor.model_validate(payload)
        else:
            descriptor = TechniqueDescriptor.seal(**payload)
        validated = advance_lifecycle(TechniqueLifecycle.DESCRIBED, TechniqueLifecycle.VALIDATED)
        registered = advance_lifecycle(validated, TechniqueLifecycle.REGISTERED)
        del registered
        return self.register(descriptor, lifecycle=TechniqueLifecycle.REGISTERED)

    def get(self, technique_id: str, version: str) -> TechniqueDescriptor:
        key = f"{technique_id}@{version}"
        descriptor = self._descriptors.get(key)
        if descriptor is None:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_UNKNOWN,
                "technique id/version is not registered",
                details={
                    "technique_id": technique_id,
                    "version": version,
                    "known": sorted(self._descriptors),
                },
            )
        return descriptor

    def resolve(self, request: TechniqueRequest) -> RuntimeAdapter:
        descriptor = self.get(request.technique_id, request.version)
        return self._adapters[descriptor.technique_key]

    def validate(
        self,
        request: TechniqueRequest,
        context: CompatibilityContext,
    ) -> str:
        adapter = self.resolve(request)
        return adapter.validate(dict(request.config), context)

    def plan(
        self,
        request: TechniqueRequest,
        context: CompatibilityContext,
    ) -> TechniquePlan:
        """Plan through the same interface callers and tests use."""
        adapter = self.resolve(request)
        return adapter.plan(dict(request.config), context)

    def list_techniques(self) -> tuple[TechniqueDescriptor, ...]:
        return tuple(self._descriptors[key] for key in sorted(self._descriptors))

    def export_descriptor(self, technique_id: str, version: str) -> dict[str, Any]:
        return self.get(technique_id, version).model_dump(mode="json")


def _adapter_for(descriptor: TechniqueDescriptor) -> RuntimeAdapter:
    if descriptor.execution is ExecutionKind.BUILTIN:
        if descriptor.technique_id == "sequence.v1":
            return BuiltinSequenceAdapter(descriptor)
        if descriptor.technique_id == "logit.v1":
            return BuiltinLogitAdapter(descriptor)
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
            "unknown builtin technique_id",
            details={"technique_id": descriptor.technique_id},
        )
    return ExternalContainerAdapter(descriptor)


def load_registry(
    *,
    external_paths: Mapping[str, Path] | None = None,
) -> TechniqueRegistry:
    """Construct a registry with builtins plus optional external descriptors."""
    registry = TechniqueRegistry.with_builtins()
    if external_paths:
        for path in external_paths.values():
            registry.register_from_path(Path(path))
    return registry


__all__ = [
    "TechniqueRegistry",
    "TechniqueRequest",
    "load_registry",
]
