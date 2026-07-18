"""Materialize portfolio slots into real ``SealedRunManifest`` contracts."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Literal

from pydantic import Field, StrictStr, model_validator

from distillery.contracts.base import FrozenDict, FrozenModel
from distillery.contracts.hashing import (
    AwareDatetime,
    PositiveSafeInt,
    Sha256Hex,
    content_sha256,
)
from distillery.contracts.manifest import (
    AutoResolverInput,
    ManifestCompletionEvidence,
    ManifestCost,
    ManifestDatasetRef,
    ManifestMemoryDryRunEvidence,
    ManifestModels,
    ManifestModelSpec,
    ManifestOutput,
    ManifestProofProtocol,
    ManifestQLoRAConfig,
    ManifestRecipe,
    ManifestRuntime,
    ManifestSpecialTokenMapEvidence,
    ManifestTraining,
    ManifestTrainingCapabilityEvidence,
    SealedRunManifest,
    manifest_capability_binding_sha256,
    manifest_length_configuration_sha256,
    manifest_memory_dry_run_evidence_sha256,
    manifest_model_configuration_sha256,
    manifest_training_configuration_sha256,
)
from experiments.portfolio.plan import (
    MemoryProbe,
    PlannedRunSlot,
    PortfolioArm,
    PortfolioPlan,
    ReadinessEvidence,
    Task,
    Tier,
    Wave,
    campaign_arm,
    validate_readiness,
)

TargetSource = Literal["oracle", "pre_materialized_teacher"]


class SlotMaterializationEvidence(FrozenModel):
    schema_version: Literal["distillery.portfolio.slot_materialization.v1"] = (
        "distillery.portfolio.slot_materialization.v1"
    )
    model_id: StrictStr
    created_at: AwareDatetime
    target_source: TargetSource
    teacher_responses_sha256: Sha256Hex | None
    completion_evidence: ManifestCompletionEvidence
    special_token_map: FrozenDict[StrictStr, PositiveSafeInt] = Field(min_length=1)
    sampler_order_hash: Sha256Hex
    source_records_sha256: Sha256Hex
    evidence_sha256: Sha256Hex

    @model_validator(mode="after")
    def _bound(self) -> SlotMaterializationEvidence:
        if (self.target_source == "pre_materialized_teacher") != (
            self.teacher_responses_sha256 is not None
        ):
            raise ValueError("teacher target source requires a sealed response hash")
        expected_label = "teacher" if self.target_source == "pre_materialized_teacher" else "oracle"
        if set(self.completion_evidence.label_source_counts) != {expected_label}:
            raise ValueError("completion evidence label source does not match treatment")
        if self.evidence_sha256 != _slot_evidence_hash(self):
            raise ValueError("slot materialization evidence hash mismatch")
        return self


def _slot_evidence_hash(value: SlotMaterializationEvidence) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"evidence_sha256"}))


def slot_materialization_evidence(**values: object) -> SlotMaterializationEvidence:
    provisional = SlotMaterializationEvidence.model_construct(
        **values,
        evidence_sha256="0" * 64,
    )
    return SlotMaterializationEvidence.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "evidence_sha256": _slot_evidence_hash(provisional),
        }
    )


def _objective_weights(plan: PortfolioPlan, arm: PortfolioArm) -> tuple[float, float]:
    if arm is PortfolioArm.LOGIT_KD:
        return (
            float(plan.protocol.logit_kd_weight),
            float(plan.protocol.logit_hard_ce_weight),
        )
    return 0.0, 1.0


def _probe_kind(arm: PortfolioArm) -> str:
    return {
        PortfolioArm.ORACLE_SFT: "oracle_sft_train_step",
        PortfolioArm.SEQUENCE_CE_CONTROL: "oracle_sft_train_step",
        PortfolioArm.SEQUENCE_KD: "sequence_kd_train_step",
        PortfolioArm.LOGIT_KD: "logit_kd_joint_train_step",
        PortfolioArm.CE_ABLATION: "ce_ablation_train_step",
    }[arm]


def _probe_for(readiness: ReadinessEvidence, arm: PortfolioArm) -> MemoryProbe:
    kind = _probe_kind(arm)
    matches = [probe for probe in readiness.probes if probe.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"readiness has no unique memory probe for {kind}")
    return matches[0]


def _price_text(hourly_microusd: int) -> str:
    return format(Decimal(hourly_microusd) / Decimal(1_000_000), "f")


def _wire_logit_memory(
    provisional: SealedRunManifest,
    probe: MemoryProbe,
) -> ManifestMemoryDryRunEvidence:
    payload: dict[str, object] = {
        "schema_version": "distillery.memory_dry_run.v2",
        "passed": True,
        "binding_sha256": manifest_capability_binding_sha256(provisional),
        "training_config_sha256": manifest_training_configuration_sha256(provisional),
        "teacher_model_config_sha256": manifest_model_configuration_sha256(
            provisional.models.teacher
        ),
        "student_model_config_sha256": manifest_model_configuration_sha256(
            provisional.models.student
        ),
        "length_config_sha256": manifest_length_configuration_sha256(provisional),
        "runtime_image_digest": provisional.runtime.image_digest,
        "instance_type": provisional.runtime.instance_type,
        "recipe_id": "logit.v1",
        "teacher_model_id": provisional.models.teacher.id,
        "teacher_revision": provisional.models.teacher.revision,
        "student_model_id": provisional.models.student.id,
        "student_revision": provisional.models.student.revision,
        "max_length": provisional.training.max_length,
        "max_completion": provisional.training.qlora.max_completion,
        "vocab_chunk_size": provisional.training.qlora.vocab_chunk,
        "peak_memory_bytes": probe.peak_bytes,
        "capacity_memory_bytes": probe.capacity_bytes,
        "headroom_bytes": probe.headroom_bytes,
        "device_type": probe.accelerator,
        "probe_id": probe.probe_sha256,
    }
    payload["evidence_sha256"] = manifest_memory_dry_run_evidence_sha256(payload)
    return ManifestMemoryDryRunEvidence.model_validate(payload)


def _slot_cost(plan: PortfolioPlan, wave: Wave, slot: PlannedRunSlot) -> int:
    matches = [wave_cost for wave_cost in plan.costs.waves if wave_cost.wave_id == wave.wave_id]
    if len(matches) != 1:
        raise ValueError("no unique cost plan for wave")
    allocations = [
        cost
        for cost in matches[0].slots
        if (cost.slot, cost.node, cost.gpu) == (slot.slot, slot.node, slot.gpu)
    ]
    if len(allocations) != 1:
        raise ValueError("no unique slot cost allocation")
    return allocations[0].allocated_ceiling_microusd


def _tier_index(tier: Tier) -> int:
    return tuple(Tier).index(tier)


def materialize_slot(
    *,
    plan: PortfolioPlan,
    wave: Wave,
    slot: PlannedRunSlot,
    readiness: ReadinessEvidence,
    evidence: SlotMaterializationEvidence,
) -> SealedRunManifest:
    """Create a fully validated manifest without launching or writing artifacts."""
    if wave not in plan.screen_waves and wave.phase != "replication":
        raise ValueError("wave is not a bound portfolio screen or replication")
    if slot not in wave.active_slots:
        raise ValueError("slot is not active in the supplied wave")
    if evidence.model_id != slot.model_id:
        raise ValueError("materialization evidence model mismatch")
    index = _tier_index(wave.tier)
    item = plan.candidates[index]
    runtime = plan.runtimes[index]
    gate = plan.gates[index]
    validate_readiness(item, runtime, gate, readiness)
    expected_target: TargetSource = (
        "pre_materialized_teacher" if slot.arm is PortfolioArm.SEQUENCE_KD else "oracle"
    )
    if evidence.target_source != expected_target:
        raise ValueError("target source differs from the declared treatment")
    if (
        evidence.completion_evidence.completion_tokenizer_sha256
        != item.pair.student.tokenizer_sha256
    ):
        raise ValueError("completion evidence is not bound to the student tokenizer")
    view = plan.dataset.view_for(slot.tasks)
    if view.view_sha256 != slot.dataset_view_sha256:
        raise ValueError("slot dataset view binding mismatch")
    price = plan.pricing[0 if wave.tier is Tier.NANO else 1]
    if price.instance_type != runtime.instance_type:
        raise ValueError("runtime pricing instance mismatch")
    max_run_microusd = _slot_cost(plan, wave, slot)
    max_run_usd = float(Decimal(max_run_microusd) / Decimal(1_000_000))
    kd_weight, hard_ce_weight = _objective_weights(plan, slot.arm)
    special_maps = ManifestSpecialTokenMapEvidence(
        teacher=evidence.special_token_map,
        student=evidence.special_token_map,
    )
    auto_input = (
        AutoResolverInput(
            cheaper_baseline_satisfies_gate=False,
            usable_responses_exist=True,
            local_white_box=True,
            tokenizer_fingerprint_match=True,
            special_token_map_match=True,
            chat_template_compatible=True,
            memory_dry_run_ok=True,
            allowed_teacher_can_fill_within_ceiling=True,
        )
        if slot.recipe == "logit.v1"
        else None
    )
    capability = ManifestTrainingCapabilityEvidence(
        special_token_maps=special_maps,
        auto_resolver_input=auto_input,
    )
    qlora = ManifestQLoRAConfig(
        rank=plan.protocol.lora_rank,
        alpha=plan.protocol.lora_alpha,
        dropout=plan.protocol.lora_dropout,
        max_completion=plan.protocol.max_completion,
        logit_temperature=plan.protocol.logit_temperature,
        kd_weight=kd_weight,
        hard_ce_weight=hard_ce_weight,
        vocab_chunk=plan.protocol.vocab_chunk,
        capability_evidence=capability,
    )
    task_view_tag = {
        "view_id": view.view_id,
        "relative_prefix": view.relative_prefix,
        "task_filter": [task.value for task in view.task_filter],
        "content_sha256": view.content_sha256,
        "split_sha256": {split.value: digest for split, digest in view.split_sha256.items()},
        "filter_protocol_sha256": view.filter_protocol_sha256,
        "view_sha256": view.view_sha256,
    }
    provisional = SealedRunManifest.model_construct(
        run_id=slot.run_id,
        created_at=evidence.created_at,
        dataset=ManifestDatasetRef(
            dataset_id=plan.dataset.bundle_id,
            uri=plan.dataset.uri,
            sha256=plan.dataset.content_sha256,
            split_sha256=plan.dataset.split_sha256,
        ),
        models=ManifestModels(
            teacher=ManifestModelSpec(
                id=item.pair.teacher.model_id,
                revision=item.pair.teacher.revision,
                tokenizer_sha256=item.pair.teacher.tokenizer_sha256,
                chat_template_sha256=item.pair.teacher.chat_template_sha256,
            ),
            student=ManifestModelSpec(
                id=item.pair.student.model_id,
                revision=item.pair.student.revision,
                tokenizer_sha256=item.pair.student.tokenizer_sha256,
                chat_template_sha256=item.pair.student.chat_template_sha256,
            ),
        ),
        recipe=ManifestRecipe(
            requested=slot.recipe,
            resolved=slot.recipe,
            resolver_reasons=("explicit_request",),
        ),
        training=ManifestTraining(
            seed=slot.seed,
            max_steps=plan.protocol.max_steps,
            token_budget=0,
            max_length=plan.protocol.max_length,
            qlora=qlora,
            completion_evidence=evidence.completion_evidence,
        ),
        proof_protocol=ManifestProofProtocol(
            id=plan.protocol.proof_protocol_id,
            sha256=plan.protocol.proof_protocol_sha256,
        ),
        runtime=ManifestRuntime(
            backend="sagemaker",
            region=price.region,
            instance_type=runtime.instance_type,
            image_digest=runtime.runtime_image_digest,
        ),
        cost=ManifestCost(
            max_run_usd=max_run_usd,
            estimate_low_usd=0.0,
            estimate_high_usd=max_run_usd,
        ),
        output=ManifestOutput(prefix=slot.output_prefix),
        package_lock_hash=runtime.package_lock_hash,
        source_revision=runtime.source_revision,
        license_dispositions={
            item.pair.teacher.model_id: (
                f"evidence_sha256:{item.pair.teacher.license_evidence_sha256}"
            ),
            item.pair.student.model_id: (
                f"evidence_sha256:{item.pair.student.license_evidence_sha256}"
            ),
        },
        tags={
            "RunMode": "portfolio-v2",
            "Arm": campaign_arm(slot.arm),
            "PortfolioArm": slot.arm.value,
            "PortfolioRole": slot.role,
            "PortfolioTier": wave.tier.value,
            "PortfolioWaveId": wave.wave_id,
            "PortfolioWaveMatrixSha256": wave.matrix_sha256,
            "PortfolioSlot": str(slot.slot),
            "PortfolioNode": str(slot.node),
            "PortfolioGpu": str(slot.gpu),
            "PortfolioModelId": slot.model_id,
            "PortfolioArtifactId": slot.artifact_id,
            "PortfolioProtocolVersion": plan.protocol.schema_version,
            "PortfolioTrainingProtocolSha256": plan.protocol.protocol_sha256,
            "PortfolioSharedProtocolSha256": wave.shared_protocol_sha256,
            "PortfolioSlotProtocolSha256": slot.protocol_sha256,
            "PortfolioGateSha256": gate.gate_sha256,
            "PortfolioMaterializationBindingSha256": (slot.materialization_binding_sha256),
            "PortfolioMaterializationEvidenceSha256": evidence.evidence_sha256,
            "PortfolioComparisonId": slot.comparison_id,
            "PortfolioComparisonPosition": slot.comparison_position,
            "PortfolioComparisonSha256": slot.comparison_sha256,
            "PortfolioTaskFilter": json.dumps(
                [task.value for task in slot.tasks],
                separators=(",", ":"),
            ),
            "PortfolioDatasetView": json.dumps(
                task_view_tag,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "PortfolioDatasetViewSha256": view.view_sha256,
            "PortfolioTaskFilterSha256": view.filter_protocol_sha256,
            "PortfolioTargetSource": evidence.target_source,
            "PortfolioSourceRecordsSha256": evidence.source_records_sha256,
            "PortfolioTeacherResponsesSha256": (evidence.teacher_responses_sha256 or "none"),
            "PortfolioReadinessSha256": readiness.readiness_sha256,
            "EnableNetworkIsolation": "true",
            "MaxRuntimeInSeconds": str(plan.protocol.max_runtime_seconds),
            "HourlyUsd": _price_text(price.current_hourly_price_microusd),
            "TrainingProtocolSha256": slot.protocol_sha256,
        },
        sampler_order_hash=evidence.sampler_order_hash,
    )
    if slot.recipe == "logit.v1":
        memory = _wire_logit_memory(provisional, _probe_for(readiness, slot.arm))
        final_capability = capability.model_copy(update={"memory_dry_run": memory})
        final_qlora = qlora.model_copy(update={"capability_evidence": final_capability})
        final_training = provisional.training.model_copy(update={"qlora": final_qlora})
        provisional = provisional.model_copy(update={"training": final_training})
    manifest = SealedRunManifest.model_validate(provisional.model_dump(mode="json"))
    validate_materialized_manifest(plan=plan, wave=wave, slot=slot, manifest=manifest)
    return manifest


def validate_materialized_manifest(
    *,
    plan: PortfolioPlan,
    wave: Wave,
    slot: PlannedRunSlot,
    manifest: SealedRunManifest,
) -> None:
    """Reject stale smoke knobs, mixed proof protocols, and task-view drift."""
    view = plan.dataset.view_for(slot.tasks)
    expected = (
        slot.run_id,
        slot.seed,
        slot.recipe,
        plan.protocol.max_steps,
        plan.protocol.max_length,
        plan.protocol.max_completion,
        plan.protocol.lora_rank,
        plan.protocol.lora_alpha,
        plan.protocol.lora_dropout,
        plan.protocol.vocab_chunk,
        plan.protocol.proof_protocol_id,
        plan.protocol.proof_protocol_sha256,
        plan.dataset.bundle_id,
        plan.dataset.content_sha256,
        wave.instance_type,
        slot.output_prefix,
        view.view_sha256,
        view.filter_protocol_sha256,
        slot.materialization_binding_sha256,
    )
    actual = (
        manifest.run_id,
        manifest.training.seed,
        manifest.recipe.resolved,
        manifest.training.max_steps,
        manifest.training.max_length,
        manifest.training.qlora.max_completion,
        manifest.training.qlora.rank,
        manifest.training.qlora.alpha,
        manifest.training.qlora.dropout,
        manifest.training.qlora.vocab_chunk,
        manifest.proof_protocol.id,
        manifest.proof_protocol.sha256,
        manifest.dataset.dataset_id,
        manifest.dataset.sha256,
        manifest.runtime.instance_type,
        manifest.output.prefix,
        manifest.tags.get("PortfolioDatasetViewSha256"),
        manifest.tags.get("PortfolioTaskFilterSha256"),
        manifest.tags.get("PortfolioMaterializationBindingSha256"),
    )
    if actual != expected:
        raise ValueError("materialized manifest differs from portfolio slot/protocol binding")
    if manifest.tags.get("RunMode") != "portfolio-v2":
        raise ValueError("portfolio manifest cannot use the v1 smoke run mode")
    if "EmergencyProfile" in manifest.tags or "ProtocolDeviation" in manifest.tags:
        raise ValueError("portfolio manifest cannot carry v1 smoke profile/deviation knobs")
    if manifest.tags.get("PortfolioProtocolVersion") != plan.protocol.schema_version:
        raise ValueError("portfolio manifest protocol version mismatch")
    if manifest.tags.get("PortfolioSharedProtocolSha256") != wave.shared_protocol_sha256:
        raise ValueError("portfolio manifest shared protocol mismatch")
    if manifest.tags.get("TrainingProtocolSha256") != slot.protocol_sha256:
        raise ValueError("portfolio manifest slot protocol mismatch")
    decoded_view = json.loads(manifest.tags["PortfolioDatasetView"])
    if decoded_view["task_filter"] != [task.value for task in slot.tasks]:
        raise ValueError("manifest task-filtered dataset does not match slot tasks")
    if slot.role == "generalist" and slot.tasks != tuple(Task):
        raise ValueError("materialized generalist must cover all four tasks")
    if slot.role == "specialist" and len(decoded_view["task_filter"]) != 1:
        raise ValueError("materialized specialist must use one explicit task filter")
