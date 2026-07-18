"""Bring Your Own Distillation Technique (BYODT) deep module.

Public seam for callers and tests:

* ``TechniqueRegistry.with_builtins()`` — resolve built-in + external techniques
* ``TechniqueRequest`` / ``CompatibilityContext`` — plan inputs
* ``registry.plan(request, context)`` — deterministic ``TechniquePlan``

External plugin code executes only inside its digest-pinned, network-isolated
container channel. The control plane never imports plugin modules.
"""

from __future__ import annotations

from distillery.techniques.builtins import (
    builtin_descriptors,
    logit_v1_descriptor,
    sequence_v1_descriptor,
)
from distillery.techniques.capabilities import (
    EvidenceRequirement,
    TechniqueCapability,
)
from distillery.techniques.channel import (
    TechniqueChannelContract,
    forbid_control_plane_import,
    load_channel_plan,
    write_channel_plan,
)
from distillery.techniques.compatibility import (
    CompatibilityContext,
    CompatibilityDecision,
    negotiate_compatibility,
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
from distillery.techniques.errors import (
    TechniqueError,
    TechniqueErrorCode,
    TechniqueErrorPayload,
)
from distillery.techniques.lifecycle import TechniqueLifecycle, advance_lifecycle
from distillery.techniques.protocol import compute_protocol_hash
from distillery.techniques.registry import (
    TechniqueRegistry,
    TechniqueRequest,
    load_registry,
)
from distillery.techniques.runtime import (
    ExternalExecutionPlan,
    LossContract,
    RuntimeAdapter,
    TechniquePlan,
)
from distillery.techniques.schema import canonical_config_hash, validate_config_against_schema

__all__ = [
    "ArtifactContract",
    "CompatibilityContext",
    "CompatibilityDecision",
    "CostModel",
    "EvidenceRequirement",
    "ExecutionKind",
    "ExternalExecutionPlan",
    "HardwareRequirements",
    "LossContract",
    "PluginImageBinding",
    "ReviewedSourceBinding",
    "RuntimeAdapter",
    "TeacherSignal",
    "TechniqueCapability",
    "TechniqueChannelContract",
    "TechniqueDescriptor",
    "TechniqueError",
    "TechniqueErrorCode",
    "TechniqueErrorPayload",
    "TechniqueLifecycle",
    "TechniquePlan",
    "TechniqueRegistry",
    "TechniqueRequest",
    "TokenizerConstraint",
    "advance_lifecycle",
    "builtin_descriptors",
    "canonical_config_hash",
    "compute_protocol_hash",
    "forbid_control_plane_import",
    "load_channel_plan",
    "load_registry",
    "logit_v1_descriptor",
    "negotiate_compatibility",
    "sequence_v1_descriptor",
    "validate_config_against_schema",
    "write_channel_plan",
]
