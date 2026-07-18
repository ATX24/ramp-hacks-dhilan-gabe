"""Gated registry promotion and publication contracts.

Building these immutable objects does not upload or publish them anywhere.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, StrictStr, model_validator

from distillery.contracts.base import FrozenModel
from distillery.contracts.hashing import (
    AwareDatetime,
    PositiveSafeInt,
    Sha256Hex,
    canonical_json_bytes,
    content_sha256,
)
from distillery.contracts.ids import RunId
from experiments.portfolio.plan import (
    ModelId,
    PlannedRegistry,
    PlannedRegistryEntry,
    PortfolioPlan,
    Task,
    Tier,
    Wave,
    _planned_registry_hash,
    descriptors,
)
from experiments.portfolio.selection import (
    MultiplicityPlan,
    SpecialistPromotionEvidence,
    specialist_eligible,
)


def _s3_uri(value: str, *, field_name: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"{field_name} must be a plain s3:// URI")
    return value


class ArtifactPublicationEvidence(FrozenModel):
    schema_version: Literal["distillery.portfolio.artifact_publication.v1"] = (
        "distillery.portfolio.artifact_publication.v1"
    )
    model_id: ModelId
    run_id: RunId
    manifest_sha256: Sha256Hex
    adapter_uri: StrictStr
    adapter_checksum_sha256: Sha256Hex
    adapter_size_bytes: PositiveSafeInt
    merged_uri: StrictStr | None = None
    merged_checksum_sha256: Sha256Hex | None = None
    merged_size_bytes: PositiveSafeInt | None = None
    artifact_inventory_sha256: Sha256Hex
    artifact_exists_evidence_sha256: Sha256Hex
    proof_report_sha256: Sha256Hex
    verified_at: AwareDatetime
    evidence_sha256: Sha256Hex

    @model_validator(mode="after")
    def _artifact(self) -> ArtifactPublicationEvidence:
        _s3_uri(self.adapter_uri, field_name="adapter URI")
        merged_values = (
            self.merged_uri,
            self.merged_checksum_sha256,
            self.merged_size_bytes,
        )
        if any(value is not None for value in merged_values):
            if any(value is None for value in merged_values):
                raise ValueError("merged export URI/checksum/size must be supplied together")
            _s3_uri(str(self.merged_uri), field_name="merged URI")
        if self.evidence_sha256 != _artifact_evidence_hash(self):
            raise ValueError("artifact publication evidence hash mismatch")
        return self


def _artifact_evidence_hash(value: ArtifactPublicationEvidence) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"evidence_sha256"}))


def artifact_publication_evidence(**values: object) -> ArtifactPublicationEvidence:
    provisional = ArtifactPublicationEvidence.model_construct(
        **values,
        evidence_sha256="0" * 64,
    )
    return ArtifactPublicationEvidence.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "evidence_sha256": _artifact_evidence_hash(provisional),
        }
    )


def register_replication_wave(
    *,
    registry: PlannedRegistry,
    plan: PortfolioPlan,
    wave: Wave,
) -> PlannedRegistry:
    if wave.phase != "replication" or wave.tier not in tuple(Tier):
        raise ValueError("only a bound replication wave can extend the planned registry")
    candidate = plan.candidates[tuple(Tier).index(wave.tier)]
    additions = descriptors(candidate, wave)
    entries = (
        *registry.entries,
        *(
            PlannedRegistryEntry(
                model_id=model.model_id,
                descriptor_sha256=model.descriptor_sha256,
                tier=model.tier,
                role=model.role,
                arm=model.arm,
                tasks=model.tasks,
                seed=model.seed,
            )
            for model in additions
        ),
    )
    provisional = PlannedRegistry.model_construct(
        entries=entries,
        registry_sha256="0" * 64,
    )
    return PlannedRegistry.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "registry_sha256": _planned_registry_hash(provisional),
        }
    )


class RegistryPromotion(FrozenModel):
    schema_version: Literal["distillery.portfolio.registry_promotion.v1"] = (
        "distillery.portfolio.registry_promotion.v1"
    )
    model_id: ModelId
    tier: Tier
    task: Task
    role: Literal["specialist"] = "specialist"
    status: Literal["specialist_backup"] = "specialist_backup"
    routing: Literal["explicit_user_switch_only"] = "explicit_user_switch_only"
    promotion_evidence_sha256: Sha256Hex
    multiplicity_plan_sha256: Sha256Hex
    artifact_evidence: ArtifactPublicationEvidence
    decision_sha256: Sha256Hex

    @model_validator(mode="after")
    def _decision(self) -> RegistryPromotion:
        if self.artifact_evidence.model_id != self.model_id:
            raise ValueError("promotion artifact model mismatch")
        if self.decision_sha256 != _promotion_hash(self):
            raise ValueError("registry promotion hash mismatch")
        return self


def _promotion_hash(value: RegistryPromotion) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"decision_sha256"}))


def promote_specialist_registry(
    *,
    plan: PortfolioPlan,
    registry: PlannedRegistry,
    replication_wave: Wave,
    multiplicity_plan: MultiplicityPlan,
    evidence: SpecialistPromotionEvidence,
    artifact: ArtifactPublicationEvidence,
) -> RegistryPromotion:
    eligible, reason = specialist_eligible(
        plan,
        replication_wave,
        multiplicity_plan,
        evidence,
    )
    if not eligible:
        raise ValueError(f"specialist is not promotion eligible: {reason}")
    model_id = evidence.replication_interval.treatment_model_id
    entries = [entry for entry in registry.entries if entry.model_id == model_id]
    if len(entries) != 1 or entries[0].role != "specialist":
        raise ValueError("specialist must remain a planned registry entry before promotion")
    slot = next(slot for slot in replication_wave.active_slots if slot.model_id == model_id)
    if (
        artifact.model_id,
        artifact.run_id,
        artifact.manifest_sha256,
        artifact.proof_report_sha256,
    ) != (
        model_id,
        slot.run_id,
        evidence.replication_treatment_manifest_sha256,
        evidence.replication_proof_report_sha256,
    ):
        raise ValueError("promotion artifact/checksum/proof evidence mismatch")
    provisional = RegistryPromotion.model_construct(
        model_id=model_id,
        tier=evidence.tier,
        task=evidence.task,
        promotion_evidence_sha256=evidence.evidence_sha256,
        multiplicity_plan_sha256=multiplicity_plan.plan_sha256,
        artifact_evidence=artifact,
        decision_sha256="0" * 64,
    )
    return RegistryPromotion.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "decision_sha256": _promotion_hash(provisional),
        }
    )


class GeneralistApproval(FrozenModel):
    schema_version: Literal["distillery.portfolio.generalist_approval.v1"] = (
        "distillery.portfolio.generalist_approval.v1"
    )
    tier: Tier
    model_id: ModelId
    proof_report_sha256: Sha256Hex
    quality_gate_evidence_sha256: Sha256Hex
    artifact_evidence: ArtifactPublicationEvidence
    approval_sha256: Sha256Hex

    @model_validator(mode="after")
    def _approval(self) -> GeneralistApproval:
        if (
            self.artifact_evidence.model_id != self.model_id
            or self.artifact_evidence.proof_report_sha256 != self.proof_report_sha256
        ):
            raise ValueError("generalist approval artifact/proof mismatch")
        if self.approval_sha256 != _generalist_approval_hash(self):
            raise ValueError("generalist approval hash mismatch")
        return self


def _generalist_approval_hash(value: GeneralistApproval) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"approval_sha256"}))


def generalist_approval(**values: object) -> GeneralistApproval:
    provisional = GeneralistApproval.model_construct(**values, approval_sha256="0" * 64)
    return GeneralistApproval.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "approval_sha256": _generalist_approval_hash(provisional),
        }
    )


class RegistryPublicationGate(FrozenModel):
    schema_version: Literal["distillery.portfolio.registry_publish_gate.v1"] = (
        "distillery.portfolio.registry_publish_gate.v1"
    )
    planned_registry_sha256: Sha256Hex
    artifact_inventory_sha256: Sha256Hex
    checksum_verifier_sha256: Sha256Hex
    proof_protocol_sha256: Sha256Hex
    published_at: AwareDatetime
    gate_evidence_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hash(self) -> RegistryPublicationGate:
        if self.gate_evidence_sha256 != _publish_gate_hash(self):
            raise ValueError("registry publication gate hash mismatch")
        return self


def _publish_gate_hash(value: RegistryPublicationGate) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"gate_evidence_sha256"}))


def registry_publication_gate(**values: object) -> RegistryPublicationGate:
    provisional = RegistryPublicationGate.model_construct(
        **values,
        gate_evidence_sha256="0" * 64,
    )
    return RegistryPublicationGate.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "gate_evidence_sha256": _publish_gate_hash(provisional),
        }
    )


class PublishedRegistryEntry(FrozenModel):
    model_id: ModelId
    tier: Tier
    role: Literal["generalist", "specialist"]
    tasks: tuple[Task, ...]
    status: Literal["tier_default", "specialist_backup"]
    routing: Literal["default_within_selected_tier", "explicit_user_switch_only"]
    adapter_uri: StrictStr
    adapter_checksum_sha256: Sha256Hex
    merged_uri: StrictStr | None
    merged_checksum_sha256: Sha256Hex | None
    manifest_sha256: Sha256Hex
    proof_report_sha256: Sha256Hex


class PublishedRegistryBundle(FrozenModel):
    schema_version: Literal["distillery.portfolio.published_registry.v1"] = (
        "distillery.portfolio.published_registry.v1"
    )
    active_default_tier: Literal[Tier.NANO] = Tier.NANO
    active_default_model_id: ModelId
    tier_default_model_ids: tuple[ModelId, ModelId, ModelId]
    entries: tuple[PublishedRegistryEntry, ...] = Field(min_length=3)
    silent_task_routing_forbidden: Literal[True] = True
    silent_tier_routing_forbidden: Literal[True] = True
    publication_gate_sha256: Sha256Hex
    bundle_sha256: Sha256Hex

    @model_validator(mode="after")
    def _bundle(self) -> PublishedRegistryBundle:
        by_id = {entry.model_id: entry for entry in self.entries}
        if len(by_id) != len(self.entries):
            raise ValueError("published registry model IDs must be unique")
        defaults = [by_id.get(model_id) for model_id in self.tier_default_model_ids]
        if any(entry is None for entry in defaults):
            raise ValueError("published registry default lacks an entry")
        if [entry.tier for entry in defaults if entry is not None] != list(Tier):
            raise ValueError("published registry requires one ordered default per tier")
        if any(
            entry is None
            or entry.role != "generalist"
            or entry.status != "tier_default"
            or entry.tasks != tuple(Task)
            for entry in defaults
        ):
            raise ValueError("tier defaults must be proved four-task generalists")
        if self.active_default_model_id != self.tier_default_model_ids[0]:
            raise ValueError("active default must start with the Nano generalist")
        if any(
            entry.role == "specialist"
            and (
                entry.status != "specialist_backup" or entry.routing != "explicit_user_switch_only"
            )
            for entry in self.entries
        ):
            raise ValueError("specialists cannot be silently routed")
        if self.bundle_sha256 != _published_registry_hash(self):
            raise ValueError("published registry bundle hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


def _published_registry_hash(value: PublishedRegistryBundle) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"bundle_sha256"}))


def publish_registry(
    *,
    plan: PortfolioPlan,
    registry: PlannedRegistry,
    gate: RegistryPublicationGate,
    generalists: tuple[GeneralistApproval, GeneralistApproval, GeneralistApproval],
    specialist_promotions: tuple[RegistryPromotion, ...] = (),
) -> PublishedRegistryBundle:
    """Build a publishable bundle only after all local evidence gates clear."""
    if (
        gate.planned_registry_sha256 != registry.registry_sha256
        or gate.proof_protocol_sha256 != plan.protocol.proof_protocol_sha256
    ):
        raise ValueError("publication gate does not bind registry and proof protocol")
    if [approval.tier for approval in generalists] != list(Tier):
        raise ValueError("generalist approvals must be ordered Nano, Core, Plus")
    planned_by_id = {entry.model_id: entry for entry in registry.entries}
    entries: list[PublishedRegistryEntry] = []
    inventory_hashes: list[str] = []
    default_ids: list[str] = []
    for approval in generalists:
        planned = planned_by_id.get(approval.model_id)
        if (
            planned is None
            or planned.role != "generalist"
            or planned.tier != approval.tier
            or planned.tasks != tuple(Task)
        ):
            raise ValueError("generalist approval does not reference a planned four-task model")
        artifact = approval.artifact_evidence
        inventory_hashes.append(artifact.artifact_inventory_sha256)
        default_ids.append(approval.model_id)
        entries.append(
            PublishedRegistryEntry(
                model_id=approval.model_id,
                tier=approval.tier,
                role="generalist",
                tasks=tuple(Task),
                status="tier_default",
                routing="default_within_selected_tier",
                adapter_uri=artifact.adapter_uri,
                adapter_checksum_sha256=artifact.adapter_checksum_sha256,
                merged_uri=artifact.merged_uri,
                merged_checksum_sha256=artifact.merged_checksum_sha256,
                manifest_sha256=artifact.manifest_sha256,
                proof_report_sha256=artifact.proof_report_sha256,
            )
        )
    for promotion in specialist_promotions:
        planned = planned_by_id.get(promotion.model_id)
        if (
            planned is None
            or planned.role != "specialist"
            or planned.tier != promotion.tier
            or planned.tasks != (promotion.task,)
        ):
            raise ValueError("specialist promotion does not reference a planned task model")
        artifact = promotion.artifact_evidence
        inventory_hashes.append(artifact.artifact_inventory_sha256)
        entries.append(
            PublishedRegistryEntry(
                model_id=promotion.model_id,
                tier=promotion.tier,
                role="specialist",
                tasks=(promotion.task,),
                status="specialist_backup",
                routing="explicit_user_switch_only",
                adapter_uri=artifact.adapter_uri,
                adapter_checksum_sha256=artifact.adapter_checksum_sha256,
                merged_uri=artifact.merged_uri,
                merged_checksum_sha256=artifact.merged_checksum_sha256,
                manifest_sha256=artifact.manifest_sha256,
                proof_report_sha256=artifact.proof_report_sha256,
            )
        )
    expected_inventory = content_sha256(sorted(inventory_hashes))
    if gate.artifact_inventory_sha256 != expected_inventory:
        raise ValueError("publication gate does not bind the exact artifact inventory")
    provisional = PublishedRegistryBundle.model_construct(
        active_default_model_id=default_ids[0],
        tier_default_model_ids=tuple(default_ids),
        entries=tuple(entries),
        publication_gate_sha256=gate.gate_evidence_sha256,
        bundle_sha256="0" * 64,
    )
    return PublishedRegistryBundle.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "bundle_sha256": _published_registry_hash(provisional),
        }
    )
