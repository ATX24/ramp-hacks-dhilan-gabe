"""Runtime adapter seam: validate/plan → existing training plan/loss contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic import Field, StrictStr

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import Sha256Hex
from distillery.techniques.compatibility import CompatibilityContext, CompatibilityDecision
from distillery.techniques.descriptor import TechniqueDescriptor
from distillery.techniques.lifecycle import TechniqueLifecycle
from distillery.training.models import TrainingLoadPlan


class LossContract(FrozenModel):
    """Stable loss/objective contract shared with Distillery trainers."""

    objective: StrictStr = Field(min_length=1)
    signal: StrictStr = Field(min_length=1)
    mode: StrictStr = Field(min_length=1)
    temperature: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    kd_weight: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    hard_ce_weight: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    fields: FrozenJsonObject = Field(default_factory=dict)


class ExternalExecutionPlan(FrozenModel):
    """Plan for executing an external technique in its pinned container."""

    execution: Literal["external_container"] = "external_container"
    image_uri: StrictStr
    image_digest: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    enable_network_isolation: Literal[True] = True
    channel_plan_filename: Literal["technique_plan.json"] = "technique_plan.json"
    reviewed_source_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    reviewed_source_tree_sha256: Sha256Hex
    import_forbidden: Literal[True] = True


class TechniquePlan(FrozenModel):
    """
    Deterministic planning result returned through the technique seam.

    Built-ins yield a ``TrainingLoadPlan`` + ``LossContract``. Externals yield
    an ``ExternalExecutionPlan`` for the network-isolated channel path.
    """

    schema_version: Literal["distillery.technique.plan.v1"] = "distillery.technique.plan.v1"
    technique_id: StrictStr
    version: StrictStr
    descriptor_sha256: Sha256Hex
    config_sha256: Sha256Hex
    protocol_sha256: Sha256Hex
    lifecycle: TechniqueLifecycle
    compatibility: CompatibilityDecision
    loss: LossContract
    training_load_plan: TrainingLoadPlan | None = None
    external_execution: ExternalExecutionPlan | None = None
    objective_fields: FrozenJsonObject = Field(default_factory=dict)
    channel_contract: FrozenJsonObject | None = None

    def plan_hash(self) -> str:
        from distillery.contracts.hashing import content_sha256

        return content_sha256(self.model_dump(mode="json"))


class RuntimeAdapter(Protocol):
    """
    Adapter interface at the technique seam.

    Implementations must be deterministic and must not silently downgrade to
    another technique. External adapters must not import plugin code.
    """

    descriptor: TechniqueDescriptor

    def validate(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> str:
        """Validate config + environment; return canonical config hash."""

    def plan(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> TechniquePlan:
        """Produce a sealed TechniquePlan through the same path callers use."""


__all__ = [
    "ExternalExecutionPlan",
    "LossContract",
    "RuntimeAdapter",
    "TechniquePlan",
]
