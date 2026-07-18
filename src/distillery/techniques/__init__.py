"""Plan-only Bring Your Own Distillation Technique (BYODT) deep module.

Public seam:

* ``TechniqueRegistry.with_builtins()`` resolves built-ins and descriptors.
* ``TechniqueRequest`` / ``CompatibilityContext`` are complete plan inputs.
* ``registry.plan(request, context)`` returns one sealed ``TechniquePlan``.
* ``recompute_protocol_hash(plan)`` independently verifies plan identity.

External techniques remain plan-only until trainer/backend consumption is
wired. Plans bind the image/source identity and isolation requirement a future
backend must enforce; this package does not claim present runtime enforcement.
"""

from __future__ import annotations

from distillery.techniques.compatibility import CompatibilityContext
from distillery.techniques.descriptor import TechniqueDescriptor
from distillery.techniques.errors import TechniqueError, TechniqueErrorCode
from distillery.techniques.protocol import recompute_protocol_hash
from distillery.techniques.registry import TechniqueRegistry, TechniqueRequest
from distillery.techniques.runtime import TechniquePlan

__all__ = [
    "CompatibilityContext",
    "TechniqueDescriptor",
    "TechniqueError",
    "TechniqueErrorCode",
    "TechniquePlan",
    "TechniqueRegistry",
    "TechniqueRequest",
    "recompute_protocol_hash",
]
