"""Immutable, plan-only TinyFable Nano/Core/Plus portfolio contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, StrictFloat, StrictStr, field_validator, model_validator

from distillery.contracts.base import FrozenModel
from distillery.contracts.hashing import (
    GitCommitSha,
    NonNegativeSafeInt,
    PositiveSafeInt,
    Sha256Hex,
    canonical_json_bytes,
    content_sha256,
)

DECISION_ID = "decision_tinyfable_tiered_portfolio_v2"
SCREEN_SEED = 17
REPLICATION_SEED = 23
SLOTS = 16
GIB = 1024**3
A10G_BYTES = 24 * GIB
A100_BYTES = 80 * GIB


class Task(StrEnum):
    TRANSACTION_REVIEW = "transaction_review"
    VARIANCE_ANALYSIS = "variance_analysis"
    MERCHANT_TAGGING = "merchant_tagging"
    CASH_RECONCILIATION = "cash_reconciliation"


TASKS: tuple[Task, ...] = tuple(Task)


class Tier(StrEnum):
    NANO = "nano"
    CORE = "core"
    PLUS = "plus"


TIERS: tuple[Tier, ...] = tuple(Tier)
Arm = Literal["oracle_sft", "sequence_kd", "logit_kd", "ce_ablation"]
Role = Literal["generalist", "specialist"]
EvidenceRole = Literal["teacher", "student"]
Recipe = Literal["sequence.v1", "logit.v1"]
Instance = Literal["ml.g5.48xlarge", "ml.p4de.24xlarge"]
Device = Literal["NVIDIA A10G", "NVIDIA A100-SXM4-80GB"]
ProbeKind = Literal[
    "oracle_sft_train_step",
    "sequence_kd_train_step",
    "logit_kd_joint_train_step",
    "ce_ablation_train_step",
    "optional_merged_export",
]
ModelId = Annotated[StrictStr, Field(pattern=r"^model_[a-z0-9][a-z0-9_-]{1,180}$")]

GENERALIST_ARMS: tuple[Arm, ...] = (
    "oracle_sft",
    "sequence_kd",
    "logit_kd",
    "ce_ablation",
)
# The task-specific CE arm is the matched hard-target control for logit KD.
# A specialist oracle arm would duplicate that signal in the 12 available slots.
SPECIALIST_ARMS: tuple[Arm, ...] = ("sequence_kd", "logit_kd", "ce_ablation")
REQUIRED_PROBES: tuple[ProbeKind, ...] = (
    "oracle_sft_train_step",
    "sequence_kd_train_step",
    "logit_kd_joint_train_step",
    "ce_ablation_train_step",
)
RECIPES: dict[Arm, Recipe] = {
    "oracle_sft": "sequence.v1",
    "sequence_kd": "sequence.v1",
    "logit_kd": "logit.v1",
    "ce_ablation": "logit.v1",
}


class ModelRoleEvidence(FrozenModel):
    schema_version: Literal["distillery.portfolio.role_evidence.v1"] = (
        "distillery.portfolio.role_evidence.v1"
    )
    role: EvidenceRole
    model_id: StrictStr
    revision: GitCommitSha
    model_config_sha256: Sha256Hex
    tokenizer_sha256: Sha256Hex
    chat_template_sha256: Sha256Hex
    special_tokens_sha256: Sha256Hex
    license_sha256: Sha256Hex
    output_use_sha256: Sha256Hex
    evidence_sha256: Sha256Hex

    @model_validator(mode="after")
    def _bound(self) -> ModelRoleEvidence:
        if self.evidence_sha256 != _role_hash(self):
            raise ValueError("role evidence hash mismatch")
        return self


def _role_hash(evidence: ModelRoleEvidence) -> str:
    body = evidence.model_dump(mode="json", exclude={"schema_version", "evidence_sha256"})
    return content_sha256({"schema_version": "distillery.portfolio.role_evidence_body.v1", **body})


def role_evidence(
    *,
    role: EvidenceRole,
    model_id: str,
    revision: str,
    model_config_sha256: str,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    special_tokens_sha256: str,
    license_sha256: str,
    output_use_sha256: str,
) -> ModelRoleEvidence:
    provisional = ModelRoleEvidence.model_construct(
        role=role,
        model_id=model_id,
        revision=revision,
        model_config_sha256=model_config_sha256,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        special_tokens_sha256=special_tokens_sha256,
        license_sha256=license_sha256,
        output_use_sha256=output_use_sha256,
        evidence_sha256="0" * 64,
    )
    return ModelRoleEvidence.model_validate(
        {**provisional.model_dump(mode="python"), "evidence_sha256": _role_hash(provisional)}
    )


class ModelPair(FrozenModel):
    teacher: ModelRoleEvidence
    student: ModelRoleEvidence
    tokenizer_parity_for_logit_kd: Literal[True] = True
    binding_sha256: Sha256Hex

    @model_validator(mode="after")
    def _roles(self) -> ModelPair:
        if self.teacher.role != "teacher" or self.student.role != "student":
            raise ValueError("pair requires role-specific teacher and student evidence")
        for name in ("tokenizer_sha256", "chat_template_sha256", "special_tokens_sha256"):
            if getattr(self.teacher, name) != getattr(self.student, name):
                raise ValueError(f"logit KD requires matching {name}")
        if self.binding_sha256 != _pair_hash(self.teacher, self.student):
            raise ValueError("model pair hash mismatch")
        return self


def _pair_hash(teacher: ModelRoleEvidence, student: ModelRoleEvidence) -> str:
    return content_sha256(
        {
            "schema_version": "distillery.portfolio.model_pair.v1",
            "teacher_evidence_sha256": teacher.evidence_sha256,
            "student_evidence_sha256": student.evidence_sha256,
            "tokenizer_parity_for_logit_kd": True,
        }
    )


def model_pair(teacher: ModelRoleEvidence, student: ModelRoleEvidence) -> ModelPair:
    return ModelPair(
        teacher=teacher,
        student=student,
        binding_sha256=_pair_hash(teacher, student),
    )


class Candidate(FrozenModel):
    tier: Tier
    name: Literal["TinyFable Nano", "TinyFable Core", "TinyFable Plus"]
    name_is_metadata_not_claim: Literal[True] = True
    priority: Literal["fastest_baseline", "priority_larger", "stretch"]
    parameter_label: Literal["494M", "1.5B", "3B"]
    pair: ModelPair
    instance_type: Instance
    hardware_profile: Literal[
        "g5-48xlarge-8xa10g-independent-v1",
        "p4de-24xlarge-8xa100-80gb-independent-v1",
    ]
    generalist_default_per_tier: Literal[True] = True
    specialist_backups_only: Literal[True] = True
    larger_is_better_assumption: Literal[False] = False
    throughput_tokens_per_second: None = None
    cost_per_1k_tokens_microusd: None = None
    descriptor_sha256: Sha256Hex

    @model_validator(mode="after")
    def _known(self) -> Candidate:
        if (
            self.name,
            self.priority,
            self.parameter_label,
            self.pair.teacher.model_id,
            self.pair.student.model_id,
            self.instance_type,
            self.hardware_profile,
        ) != _tier_spec(self.tier):
            raise ValueError("tier/model-role/hardware mismatch")
        if self.descriptor_sha256 != _candidate_hash(self):
            raise ValueError("candidate hash mismatch")
        return self


def _tier_spec(tier: Tier) -> tuple[str, ...]:
    return {
        Tier.NANO: (
            "TinyFable Nano",
            "fastest_baseline",
            "494M",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "ml.g5.48xlarge",
            "g5-48xlarge-8xa10g-independent-v1",
        ),
        Tier.CORE: (
            "TinyFable Core",
            "priority_larger",
            "1.5B",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "ml.p4de.24xlarge",
            "p4de-24xlarge-8xa100-80gb-independent-v1",
        ),
        Tier.PLUS: (
            "TinyFable Plus",
            "stretch",
            "3B",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "ml.p4de.24xlarge",
            "p4de-24xlarge-8xa100-80gb-independent-v1",
        ),
    }[tier]


def _candidate_hash(candidate: Candidate) -> str:
    body = candidate.model_dump(mode="json", exclude={"descriptor_sha256"})
    return content_sha256({"schema_version": "distillery.portfolio.candidate.v1", **body})


def candidate(tier: Tier, pair: ModelPair) -> Candidate:
    name, priority, params, _, _, instance, profile = _tier_spec(tier)
    provisional = Candidate.model_construct(
        tier=tier,
        name=name,
        priority=priority,
        parameter_label=params,
        pair=pair,
        instance_type=instance,
        hardware_profile=profile,
        descriptor_sha256="0" * 64,
    )
    return Candidate.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "descriptor_sha256": _candidate_hash(provisional),
        }
    )


class ProtocolGate(FrozenModel):
    tier: Tier
    candidate_sha256: Sha256Hex
    pair_sha256: Sha256Hex
    instance_type: Instance
    device: Device
    capacity_bytes: PositiveSafeInt
    max_peak_basis_points: Literal[8500] = 8500
    min_headroom_bytes: PositiveSafeInt
    required_probes: tuple[ProbeKind, ...] = REQUIRED_PROBES
    optional_merge_requires_probe: Literal[True] = True
    max_length: Literal[512] = 512
    max_completion: Literal[128] = 128
    microbatch: Literal[1] = 1
    grad_accumulation: Literal[1] = 1
    lora_rank: Literal[8] = 8
    lora_alpha: Literal[16] = 16
    lora_dropout_basis_points: Literal[500] = 500
    student_precision: Literal["qlora_nf4"] = "qlora_nf4"
    logit_teacher_precision: Literal["bf16_frozen_no_grad"] = "bf16_frozen_no_grad"
    vocab_chunk: Literal[4096] = 4096
    deterministic: Literal[True] = True
    network_isolation: Literal[True] = True
    tokenizer_license_output_gates_required: Literal[True] = True
    measured_probe_required: Literal[True] = True
    estimates_cannot_pass: Literal[True] = True
    gate_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hardware(self) -> ProtocolGate:
        expected = (
            ("ml.g5.48xlarge", "NVIDIA A10G", A10G_BYTES, 4 * GIB)
            if self.tier == Tier.NANO
            else ("ml.p4de.24xlarge", "NVIDIA A100-SXM4-80GB", A100_BYTES, 8 * GIB)
        )
        if (
            self.instance_type,
            self.device,
            self.capacity_bytes,
            self.min_headroom_bytes,
        ) != expected:
            raise ValueError("exact hardware memory gate mismatch")
        if self.gate_sha256 != _gate_hash(self):
            raise ValueError("protocol gate hash mismatch")
        return self


def _gate_hash(gate: ProtocolGate) -> str:
    body = gate.model_dump(mode="json", exclude={"gate_sha256"})
    return content_sha256({"schema_version": "distillery.portfolio.gate.v1", **body})


def protocol_gate(item: Candidate) -> ProtocolGate:
    device, capacity, headroom = (
        ("NVIDIA A10G", A10G_BYTES, 4 * GIB)
        if item.tier == Tier.NANO
        else ("NVIDIA A100-SXM4-80GB", A100_BYTES, 8 * GIB)
    )
    provisional = ProtocolGate.model_construct(
        tier=item.tier,
        candidate_sha256=item.descriptor_sha256,
        pair_sha256=item.pair.binding_sha256,
        instance_type=item.instance_type,
        device=device,
        capacity_bytes=capacity,
        min_headroom_bytes=headroom,
        gate_sha256="0" * 64,
    )
    return ProtocolGate.model_validate(
        {**provisional.model_dump(mode="python"), "gate_sha256": _gate_hash(provisional)}
    )


class MemoryProbe(FrozenModel):
    kind: ProbeKind
    passed: Literal[True] = True
    candidate_sha256: Sha256Hex
    pair_sha256: Sha256Hex
    gate_sha256: Sha256Hex
    instance_type: Instance
    device: Device
    runtime_image_digest: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    peak_bytes: PositiveSafeInt
    capacity_bytes: PositiveSafeInt
    headroom_bytes: NonNegativeSafeInt
    manifest_sha256: Sha256Hex

    @model_validator(mode="after")
    def _arithmetic(self) -> MemoryProbe:
        if self.headroom_bytes != self.capacity_bytes - self.peak_bytes:
            raise ValueError("memory headroom arithmetic mismatch")
        return self


class ReadinessEvidence(FrozenModel):
    tier: Tier
    gate_sha256: Sha256Hex
    role_evidence_verified: Literal[True] = True
    tokenizer_compatible: Literal[True] = True
    licenses_approved: Literal[True] = True
    output_use_approved: Literal[True] = True
    probes: tuple[MemoryProbe, ...] = Field(min_length=4, max_length=5)
    manifest_sha256: Sha256Hex


def validate_readiness(
    item: Candidate,
    gate: ProtocolGate,
    evidence: ReadinessEvidence,
    *,
    merge_requested: bool = False,
) -> None:
    if (item.tier, gate.tier, evidence.tier) != (gate.tier, gate.tier, gate.tier):
        raise ValueError("readiness tier mismatch")
    if evidence.gate_sha256 != gate.gate_sha256:
        raise ValueError("readiness gate hash mismatch")
    required = set(REQUIRED_PROBES)
    if merge_requested:
        required.add("optional_merged_export")
    kinds = [probe.kind for probe in evidence.probes]
    if len(kinds) != len(set(kinds)) or set(kinds) != required:
        raise ValueError("memory probes do not exactly cover required paths")
    max_peak = gate.capacity_bytes * 8500 // 10_000
    for probe in evidence.probes:
        if (
            probe.candidate_sha256 != item.descriptor_sha256
            or probe.pair_sha256 != item.pair.binding_sha256
            or probe.gate_sha256 != gate.gate_sha256
        ):
            raise ValueError("probe binding mismatch")
        if (probe.instance_type, probe.device, probe.capacity_bytes) != (
            gate.instance_type,
            gate.device,
            gate.capacity_bytes,
        ):
            raise ValueError("probe hardware mismatch")
        if probe.peak_bytes > max_peak:
            raise ValueError("measured peak exceeds 85% ceiling")
        if probe.headroom_bytes < gate.min_headroom_bytes:
            raise ValueError("measured headroom below exact minimum")


class Slot(FrozenModel):
    slot: NonNegativeSafeInt = Field(le=15)
    node: Literal[0, 1]
    gpu: NonNegativeSafeInt = Field(le=7)
    run_id: StrictStr = Field(pattern=r"^run_portfolio_[a-z0-9_-]{1,160}$")
    role: Role
    arm: Arm
    tasks: tuple[Task, ...] = Field(min_length=1)
    recipe: Recipe
    seed: PositiveSafeInt
    matched_control_of: Arm | None
    rationale: StrictStr
    artifact_suffix: StrictStr
    protocol_sha256: Sha256Hex
    manifest_binding_sha256: Sha256Hex

    @model_validator(mode="after")
    def _fair(self) -> Slot:
        if (self.node, self.gpu) != (self.slot % 2, self.slot // 2):
            raise ValueError("slot must use deterministic round-robin assignment")
        if self.recipe != RECIPES[self.arm]:
            raise ValueError("recipe/arm mismatch")
        if (self.arm == "ce_ablation") != (self.matched_control_of == "logit_kd"):
            raise ValueError("CE must be the declared logit control")
        return self


class Wave(FrozenModel):
    wave_id: StrictStr = Field(pattern=r"^wave_[a-z0-9][a-z0-9_-]{1,126}$")
    tier: Tier
    candidate_sha256: Sha256Hex
    pair_sha256: Sha256Hex
    gate_sha256: Sha256Hex
    instance_type: Instance
    hardware_profile: StrictStr
    seed: PositiveSafeInt
    slots: tuple[Slot, ...] = Field(min_length=16, max_length=16)
    matrix_sha256: Sha256Hex

    @model_validator(mode="after")
    def _complete(self) -> Wave:
        if [slot.slot for slot in self.slots] != list(range(16)):
            raise ValueError("slots must be ordered 0..15")
        if any(slot.seed != self.seed for slot in self.slots):
            raise ValueError("all matched arms require the same seed")
        for field in ("run_id", "artifact_suffix", "manifest_binding_sha256"):
            values = [getattr(slot, field) for slot in self.slots]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {field}")
        generalists = [slot for slot in self.slots if slot.role == "generalist"]
        specialists = [slot for slot in self.slots if slot.role == "specialist"]
        if {slot.arm for slot in generalists} != set(GENERALIST_ARMS):
            raise ValueError("four matched generalist arms required")
        for task in TASKS:
            task_slots = [slot for slot in specialists if slot.tasks == (task,)]
            if {slot.arm for slot in task_slots} != set(SPECIALIST_ARMS):
                raise ValueError(f"specialist treatments/controls missing for {task.value}")
        if self.matrix_sha256 != _wave_hash(self):
            raise ValueError("matrix hash mismatch")
        return self


def _wave_hash(wave: Wave) -> str:
    body = wave.model_dump(mode="json", exclude={"matrix_sha256"})
    return content_sha256({"schema_version": "distillery.portfolio.wave.v1", **body})


def _slot_rationale(role: Role, arm: Arm, task: Task | None) -> str:
    if role == "generalist":
        return {
            "oracle_sft": (
                "Shared four-task hard-CE baseline; initial tier default, not a size claim."
            ),
            "sequence_kd": (
                "Shared sequence KD with matched seed, mixture, QLoRA, and validation."
            ),
            "logit_kd": "Shared forward-KL treatment matched directly to CE ablation.",
            "ce_ablation": "Shared hard-CE control on the logit.v1 trainer surface.",
        }[arm]
    label = task.value if task else "unknown"
    return {
        "sequence_kd": f"{label} sequence-KD specialist; explicit backup only.",
        "logit_kd": f"{label} logit-KD specialist matched to task-specific CE.",
        "ce_ablation": f"{label} hard-CE control and non-KD specialist baseline.",
    }[arm]


def wave(item: Candidate, gate: ProtocolGate, wave_id: str) -> Wave:
    if gate.candidate_sha256 != item.descriptor_sha256:
        raise ValueError("gate/candidate mismatch")
    result: list[Slot] = []
    prefix = "w1" if item.tier == Tier.NANO else wave_id.removeprefix("wave_")
    assignments: list[tuple[Role, Arm, tuple[Task, ...], Task | None]] = [
        ("generalist", arm, TASKS, None) for arm in GENERALIST_ARMS
    ] + [("specialist", arm, (task,), task) for task in TASKS for arm in SPECIALIST_ARMS]
    for index, (role, arm, tasks, task) in enumerate(assignments):
        task_slug = task.value.replace("_", "-") if task else ""
        run_id = (
            f"run_portfolio_{prefix}_generalist_{arm}"
            if role == "generalist"
            else f"run_portfolio_{prefix}_specialist_{task_slug}_{arm}"
        )
        protocol = content_sha256(
            {
                "schema_version": "distillery.portfolio.slot_protocol.v1",
                "tier": item.tier.value,
                "candidate_sha256": item.descriptor_sha256,
                "pair_sha256": item.pair.binding_sha256,
                "gate_sha256": gate.gate_sha256,
                "wave_id": wave_id,
                "role": role,
                "arm": arm,
                "tasks": [value.value for value in tasks],
                "recipe": RECIPES[arm],
                "seed": SCREEN_SEED,
                "throughput_tokens_per_second": None,
                "test_set_selection": False,
            }
        )
        result.append(
            Slot(
                slot=index,
                node=index % 2,  # type: ignore[arg-type]
                gpu=index // 2,
                run_id=run_id,
                role=role,
                arm=arm,
                tasks=tasks,
                recipe=RECIPES[arm],
                seed=SCREEN_SEED,
                matched_control_of="logit_kd" if arm == "ce_ablation" else None,
                rationale=_slot_rationale(role, arm, task),
                artifact_suffix=f"{item.tier.value}/{wave_id}/{role}/{task_slug}/{arm}/",
                protocol_sha256=protocol,
                manifest_binding_sha256=content_sha256(
                    {"run_id": run_id, "protocol_sha256": protocol, "slot": index}
                ),
            )
        )
    provisional = Wave.model_construct(
        wave_id=wave_id,
        tier=item.tier,
        candidate_sha256=item.descriptor_sha256,
        pair_sha256=item.pair.binding_sha256,
        gate_sha256=gate.gate_sha256,
        instance_type=item.instance_type,
        hardware_profile=item.hardware_profile,
        seed=SCREEN_SEED,
        slots=tuple(result),
        matrix_sha256="0" * 64,
    )
    return Wave.model_validate(
        {**provisional.model_dump(mode="python"), "matrix_sha256": _wave_hash(provisional)}
    )


class PromotionRules(FrozenModel):
    screen_seed: Literal[17] = SCREEN_SEED
    replication_seed: Literal[23] = REPLICATION_SEED
    same_protocol_except_seed: Literal[True] = True
    validation_selection_only: Literal[True] = True
    test_selection_forbidden: Literal[True] = True
    generalist_default_per_tier: Literal[True] = True
    silent_routing_forbidden: Literal[True] = True
    specialist_quality_delta: StrictFloat = 0.02
    quality_led_delta: StrictFloat = 0.02
    quality_led_min_throughput_ratio: StrictFloat = 0.80
    quality_led_max_cost_ratio: StrictFloat = 1.25
    efficiency_noninferiority: StrictFloat = -0.005
    efficiency_min_throughput_ratio: StrictFloat = 1.10
    efficiency_max_cost_ratio: StrictFloat = 0.90
    bootstrap_samples: Literal[2000] = 2000
    no_size_prior: Literal[True] = True
    throughput_and_cost_must_be_measured: Literal[True] = True


class SpecialistEvidence(FrozenModel):
    tier: Tier
    task: Task
    quality_delta: StrictFloat
    quality_ci_lower: StrictFloat
    quality_ci_upper: StrictFloat
    validation_protocol_sha256: Sha256Hex
    used_test_set: Literal[False] = False
    silent_routing: Literal[False] = False


def specialist_eligible(evidence: SpecialistEvidence, rules: PromotionRules) -> tuple[bool, str]:
    eligible = (
        evidence.quality_delta >= rules.specialist_quality_delta
        and evidence.quality_ci_lower >= rules.specialist_quality_delta
        and evidence.quality_ci_lower <= evidence.quality_ci_upper
    )
    return (
        (True, "explicit backup eligible; generalist remains default")
        if eligible
        else (False, "material gain after uncertainty not established")
    )


class TierEvidence(FrozenModel):
    candidate: Tier
    incumbent: Tier
    quality_ci_lower: StrictFloat
    quality_ci_upper: StrictFloat
    throughput_ratio_ci_lower: StrictFloat = Field(gt=0.0)
    throughput_ratio_ci_upper: StrictFloat = Field(gt=0.0)
    cost_ratio_ci_lower: StrictFloat = Field(gt=0.0)
    cost_ratio_ci_upper: StrictFloat = Field(gt=0.0)
    measurement_protocol_sha256: Sha256Hex
    same_harness_and_tokens: Literal[True] = True
    used_test_set: Literal[False] = False


def tier_eligible(evidence: TierEvidence, rules: PromotionRules) -> tuple[bool, str]:
    if (
        evidence.quality_ci_lower > evidence.quality_ci_upper
        or evidence.throughput_ratio_ci_lower > evidence.throughput_ratio_ci_upper
        or evidence.cost_ratio_ci_lower > evidence.cost_ratio_ci_upper
    ):
        return False, "invalid confidence interval"
    quality = (
        evidence.quality_ci_lower >= rules.quality_led_delta
        and evidence.throughput_ratio_ci_lower >= rules.quality_led_min_throughput_ratio
        and evidence.cost_ratio_ci_upper <= rules.quality_led_max_cost_ratio
    )
    efficiency = (
        evidence.quality_ci_lower >= rules.efficiency_noninferiority
        and evidence.throughput_ratio_ci_lower >= rules.efficiency_min_throughput_ratio
        and evidence.cost_ratio_ci_upper <= rules.efficiency_max_cost_ratio
    )
    return (
        (True, "pre-registered quality/efficiency tradeoff clears")
        if quality or efficiency
        else (False, "candidate clears neither pre-registered tradeoff path")
    )


class ModelDescriptor(FrozenModel):
    model_id: ModelId
    tier: Tier
    candidate_name: StrictStr
    role: Role
    arm: Arm
    tasks: tuple[Task, ...]
    student_model_id: StrictStr
    student_revision: GitCommitSha
    teacher_model_id: StrictStr
    teacher_revision: GitCommitSha
    pair_sha256: Sha256Hex
    recipe: Recipe
    seed: Literal[17] = SCREEN_SEED
    protocol_sha256: Sha256Hex
    manifest_binding_sha256: Sha256Hex
    artifact_kind: Literal["peft_qlora_adapter"] = "peft_qlora_adapter"
    optional_merged_export: Literal[True] = True
    status: Literal["tier_default", "generalist_candidate", "specialist_backup"]
    explicit_switch_only: Literal[True] = True
    throughput_tokens_per_second: None = None
    cost_per_1k_tokens_microusd: None = None


def descriptors(item: Candidate, matrix: Wave) -> tuple[ModelDescriptor, ...]:
    values: list[ModelDescriptor] = []
    for slot in matrix.slots:
        task = slot.tasks[0].value if slot.role == "specialist" else ""
        model_id = (
            f"model_tinyfable_{item.tier.value}_generalist_{slot.arm}"
            if slot.role == "generalist"
            else f"model_tinyfable_{item.tier.value}_specialist_{task}_{slot.arm}"
        )
        status = (
            "tier_default"
            if slot.role == "generalist" and slot.arm == "oracle_sft"
            else ("generalist_candidate" if slot.role == "generalist" else "specialist_backup")
        )
        values.append(
            ModelDescriptor(
                model_id=model_id,
                tier=item.tier,
                candidate_name=item.name,
                role=slot.role,
                arm=slot.arm,
                tasks=slot.tasks,
                student_model_id=item.pair.student.model_id,
                student_revision=item.pair.student.revision,
                teacher_model_id=item.pair.teacher.model_id,
                teacher_revision=item.pair.teacher.revision,
                pair_sha256=item.pair.binding_sha256,
                recipe=slot.recipe,
                protocol_sha256=slot.protocol_sha256,
                manifest_binding_sha256=slot.manifest_binding_sha256,
                status=status,  # type: ignore[arg-type]
            )
        )
    return tuple(values)


class Artifact(FrozenModel):
    model_id: ModelId
    adapter_uri: StrictStr
    merged_uri_optional: StrictStr
    manifest_binding_sha256: Sha256Hex
    protocol_sha256: Sha256Hex
    mutate_v1_forbidden: Literal[True] = True


class ArtifactPlan(FrozenModel):
    root: StrictStr
    artifacts: tuple[Artifact, ...] = Field(min_length=48, max_length=48)

    @field_validator("root")
    @classmethod
    def _root(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "s3" or not parsed.netloc or not value.endswith("/"):
            raise ValueError("artifact root must be an s3:// prefix ending '/'")
        return value

    @model_validator(mode="after")
    def _isolated(self) -> ArtifactPlan:
        for field in ("model_id", "adapter_uri", "merged_uri_optional"):
            values = [getattr(item, field) for item in self.artifacts]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate artifact {field}")
        return self


class Pricing(FrozenModel):
    instance_type: Instance
    hourly_microusd: PositiveSafeInt
    evidence_sha256: Sha256Hex


class SlotCost(FrozenModel):
    run_id: StrictStr
    node: Literal[0, 1]
    ceiling_microusd: NonNegativeSafeInt


class WaveCost(FrozenModel):
    tier: Tier
    parent_ceiling_microusd: NonNegativeSafeInt
    aggregate_ceiling_microusd: NonNegativeSafeInt
    slots: tuple[SlotCost, ...] = Field(min_length=16, max_length=16)
    throughput_tokens_per_second: None = None
    cost_per_1k_tokens_microusd: None = None

    @model_validator(mode="after")
    def _sum(self) -> WaveCost:
        if sum(item.ceiling_microusd for item in self.slots) != (self.aggregate_ceiling_microusd):
            raise ValueError("cost allocation mismatch")
        for node in (0, 1):
            if (
                sum(item.ceiling_microusd for item in self.slots if item.node == node)
                != self.parent_ceiling_microusd
            ):
                raise ValueError("per-node cost allocation mismatch")
        return self


def cost(matrix: Wave, price: Pricing, runtime_seconds: int = 900) -> WaveCost:
    if matrix.instance_type != price.instance_type:
        raise ValueError("pricing hardware mismatch")
    parent = (price.hourly_microusd * runtime_seconds + 3599) // 3600
    base, remainder = divmod(parent, 8)
    slots = tuple(
        SlotCost(
            run_id=item.run_id,
            node=item.node,
            ceiling_microusd=base + (1 if item.gpu < remainder else 0),
        )
        for item in matrix.slots
    )
    return WaveCost(
        tier=matrix.tier,
        parent_ceiling_microusd=parent,
        aggregate_ceiling_microusd=parent * 2,
        slots=slots,
    )


class RegistryEntry(FrozenModel):
    model_id: ModelId
    tier: Tier
    role: Role
    status: Literal["tier_default", "generalist_candidate", "specialist_backup"]
    tasks: tuple[Task, ...]
    adapter_uri: StrictStr
    merged_uri_optional: StrictStr
    explicit_switch_only: Literal[True] = True
    throughput_tokens_per_second: None = None


class Registry(FrozenModel):
    active_default_tier: Literal[Tier.NANO] = Tier.NANO
    active_default_model_id: ModelId
    tier_default_model_ids: tuple[ModelId, ModelId, ModelId]
    entries: tuple[RegistryEntry, ...] = Field(min_length=48, max_length=48)
    silent_task_routing_forbidden: Literal[True] = True
    silent_tier_routing_forbidden: Literal[True] = True
    larger_is_better_assumption: Literal[False] = False
    bundle_sha256: Sha256Hex

    @model_validator(mode="after")
    def _defaults(self) -> Registry:
        by_id = {entry.model_id: entry for entry in self.entries}
        defaults = [by_id[value] for value in self.tier_default_model_ids]
        if [item.tier for item in defaults] != list(TIERS):
            raise ValueError("one ordered generalist default required per tier")
        if any(item.role != "generalist" or item.status != "tier_default" for item in defaults):
            raise ValueError("tier defaults must be generalists")
        if self.active_default_model_id != self.tier_default_model_ids[0]:
            raise ValueError("active default starts with Nano generalist")
        if self.bundle_sha256 != _registry_hash(self):
            raise ValueError("registry hash mismatch")
        return self


def _registry_hash(registry: Registry) -> str:
    body = registry.model_dump(mode="json", exclude={"bundle_sha256"})
    return content_sha256({"schema_version": "distillery.portfolio.registry.v2", **body})


class PortfolioPlan(FrozenModel):
    schema_version: Literal["distillery.portfolio.plan.v2"] = "distillery.portfolio.plan.v2"
    decision_id: Literal["decision_tinyfable_tiered_portfolio_v2"] = DECISION_ID
    supersedes: Literal["one_generalist_only_2026-07-18"] = "one_generalist_only_2026-07-18"
    mode: Literal["plan_only"] = "plan_only"
    candidates: tuple[Candidate, Candidate, Candidate]
    gates: tuple[ProtocolGate, ProtocolGate, ProtocolGate]
    waves: tuple[Wave, Wave, Wave]
    promotion: PromotionRules
    models: tuple[ModelDescriptor, ...] = Field(min_length=48, max_length=48)
    artifacts: ArtifactPlan
    costs: tuple[WaveCost, WaveCost, WaveCost]
    registry: Registry
    plan_sha256: Sha256Hex

    @model_validator(mode="after")
    def _complete(self) -> PortfolioPlan:
        if [value.tier for value in self.candidates] != list(TIERS):
            raise ValueError("candidate order must be Nano, Core, Plus")
        if any(len([model for model in self.models if model.tier == tier]) != 16 for tier in TIERS):
            raise ValueError("each tier requires sixteen model descriptors")
        if self.plan_sha256 != _plan_hash(self):
            raise ValueError("portfolio plan hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


def _plan_hash(plan: PortfolioPlan) -> str:
    body = plan.model_dump(mode="json", exclude={"plan_sha256"})
    return content_sha256({"schema_version": "distillery.portfolio.plan_body.v2", **body})


def build_plan(
    *,
    nano_pair: ModelPair,
    core_pair: ModelPair,
    plus_pair: ModelPair,
    g5_pricing: Pricing,
    p4de_pricing: Pricing,
    artifact_root: str,
) -> PortfolioPlan:
    candidates = (
        candidate(Tier.NANO, nano_pair),
        candidate(Tier.CORE, core_pair),
        candidate(Tier.PLUS, plus_pair),
    )
    gates = tuple(protocol_gate(value) for value in candidates)
    waves = (
        wave(candidates[0], gates[0], "wave_portfolio_w1_nano"),
        wave(candidates[1], gates[1], "wave_core_a100_screen_v1"),
        wave(candidates[2], gates[2], "wave_plus_a100_screen_v1"),
    )
    models = tuple(
        model
        for item, matrix in zip(candidates, waves, strict=True)
        for model in descriptors(item, matrix)
    )
    root = artifact_root.rstrip("/") + "/"
    by_protocol = {model.protocol_sha256: model for model in models}
    artifacts = ArtifactPlan(
        root=root,
        artifacts=tuple(
            Artifact(
                model_id=by_protocol[slot.protocol_sha256].model_id,
                adapter_uri=f"{root}{slot.artifact_suffix}adapter/",
                merged_uri_optional=f"{root}{slot.artifact_suffix}merged_optional/",
                manifest_binding_sha256=slot.manifest_binding_sha256,
                protocol_sha256=slot.protocol_sha256,
            )
            for matrix in waves
            for slot in matrix.slots
        ),
    )
    artifact_by_model = {item.model_id: item for item in artifacts.artifacts}
    entries = tuple(
        RegistryEntry(
            model_id=model.model_id,
            tier=model.tier,
            role=model.role,
            status=model.status,
            tasks=model.tasks,
            adapter_uri=artifact_by_model[model.model_id].adapter_uri,
            merged_uri_optional=artifact_by_model[model.model_id].merged_uri_optional,
        )
        for model in models
    )
    default_ids = tuple(
        next(
            entry.model_id
            for entry in entries
            if entry.tier == tier and entry.status == "tier_default"
        )
        for tier in TIERS
    )
    registry_provisional = Registry.model_construct(
        active_default_model_id=default_ids[0],
        tier_default_model_ids=default_ids,
        entries=entries,
        bundle_sha256="0" * 64,
    )
    registry = Registry.model_validate(
        {
            **registry_provisional.model_dump(mode="python"),
            "bundle_sha256": _registry_hash(registry_provisional),
        }
    )
    provisional = PortfolioPlan.model_construct(
        candidates=candidates,
        gates=gates,
        waves=waves,
        promotion=PromotionRules(),
        models=models,
        artifacts=artifacts,
        costs=(
            cost(waves[0], g5_pricing),
            cost(waves[1], p4de_pricing),
            cost(waves[2], p4de_pricing),
        ),
        registry=registry,
        plan_sha256="0" * 64,
    )
    return PortfolioPlan.model_validate(
        {**provisional.model_dump(mode="python"), "plan_sha256": _plan_hash(provisional)}
    )
