"""Exact technique resolution, lifecycle, and plan-only orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, StrictStr

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import content_sha256
from distillery.techniques.adapters.builtin import (
    BuiltinLogitAdapter,
    BuiltinSequenceAdapter,
)
from distillery.techniques.adapters.external import ExternalContainerAdapter
from distillery.techniques.builtins import builtin_descriptors
from distillery.techniques.compatibility import CompatibilityContext
from distillery.techniques.descriptor import (
    RESERVED_BUILTIN_IDS,
    ExecutionKind,
    TechniqueDescriptor,
)
from distillery.techniques.errors import (
    TechniqueError,
    TechniqueErrorCode,
    raise_technique_error,
)
from distillery.techniques.lifecycle import TechniqueLifecycle, advance_lifecycle
from distillery.techniques.runtime import RuntimeAdapter, TechniquePlan


class TechniqueRequest(FrozenModel):
    technique_id: StrictStr = Field(min_length=1)
    version: StrictStr = Field(min_length=1)
    config: FrozenJsonObject = Field(default_factory=dict)


class TechniqueRegistry:
    """Deep seam: register immutable descriptors and produce sealed plans."""

    def __init__(self) -> None:
        self._descriptors: dict[str, TechniqueDescriptor] = {}
        self._adapters: dict[str, RuntimeAdapter] = {}
        self._registration_history: dict[str, tuple[TechniqueLifecycle, ...]] = {}
        self._plan_states: dict[str, TechniqueLifecycle] = {}
        self._plans: dict[str, TechniquePlan] = {}

    @classmethod
    def with_builtins(cls) -> TechniqueRegistry:
        registry = cls()
        for descriptor in builtin_descriptors():
            registry._store_descriptor(descriptor)
        return registry

    def register(self, descriptor: TechniqueDescriptor) -> TechniqueDescriptor:
        descriptor.assert_integrity()
        canonical_builtins = {item.technique_id: item for item in builtin_descriptors()}
        if descriptor.technique_id in RESERVED_BUILTIN_IDS:
            canonical = canonical_builtins[descriptor.technique_id]
            if (
                descriptor.technique_key != canonical.technique_key
                or descriptor.descriptor_sha256 != canonical.descriptor_sha256
            ):
                raise_technique_error(
                    TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                    "built-in technique IDs are reserved across all versions",
                    details={
                        "technique_id": descriptor.technique_id,
                        "version": descriptor.version,
                    },
                )
        existing = self._descriptors.get(descriptor.technique_key)
        if existing is not None:
            if existing.descriptor_sha256 == descriptor.descriptor_sha256:
                return existing
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_VERSION_COLLISION,
                "divergent descriptor already registered for technique id/version",
                details={
                    "technique_id": descriptor.technique_id,
                    "version": descriptor.version,
                    "existing_descriptor_sha256": existing.descriptor_sha256,
                    "incoming_descriptor_sha256": descriptor.descriptor_sha256,
                },
            )
        self._store_descriptor(descriptor)
        return descriptor

    def _store_descriptor(self, descriptor: TechniqueDescriptor) -> None:
        described = TechniqueLifecycle.DESCRIBED
        validated = advance_lifecycle(described, TechniqueLifecycle.VALIDATED)
        registered = advance_lifecycle(validated, TechniqueLifecycle.REGISTERED)
        key = descriptor.technique_key
        self._descriptors[key] = descriptor
        self._adapters[key] = _adapter_for(descriptor)
        self._registration_history[key] = (described, validated, registered)

    def register_from_path(self, path: Path) -> TechniqueDescriptor:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "technique descriptor file must contain a JSON object",
            )
        descriptor = (
            TechniqueDescriptor.model_validate(payload)
            if "descriptor_sha256" in payload
            else TechniqueDescriptor.seal(**payload)
        )
        return self.register(descriptor)

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

    def plan(
        self,
        request: TechniqueRequest,
        context: CompatibilityContext,
    ) -> TechniquePlan:
        """Complete config + compatibility preflight and sealed planning."""
        descriptor = self.get(request.technique_id, request.version)
        adapter = self._adapters[descriptor.technique_key]
        plan_key = content_sha256(
            {
                "technique_id": request.technique_id,
                "version": request.version,
                "config": dict(request.config),
                "context": context.model_dump(mode="json"),
            }
        )
        cached = self._plans.get(plan_key)
        if cached is not None:
            return cached
        try:
            plan = adapter.plan(dict(request.config), context)
        except TechniqueError:
            self._plan_states[plan_key] = advance_lifecycle(
                TechniqueLifecycle.REGISTERED,
                TechniqueLifecycle.REJECTED,
            )
            raise
        compatible = advance_lifecycle(
            TechniqueLifecycle.REGISTERED,
            TechniqueLifecycle.COMPATIBLE,
        )
        self._plan_states[plan_key] = compatible
        planned = advance_lifecycle(compatible, TechniqueLifecycle.PLANNED)
        expected_history = (
            TechniqueLifecycle.REGISTERED,
            TechniqueLifecycle.COMPATIBLE,
            TechniqueLifecycle.PLANNED,
        )
        if plan.lifecycle is not planned or plan.lifecycle_history != expected_history:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_LIFECYCLE_INVALID,
                "adapter returned invalid lifecycle history",
            )
        self._plan_states[plan_key] = planned
        self._plans[plan_key] = plan
        return plan

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


__all__ = ["TechniqueRegistry", "TechniqueRequest"]
