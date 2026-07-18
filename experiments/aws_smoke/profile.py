"""Fixed emergency training profile: 5–10 steps, tiny corpus, 15-minute ceiling."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from experiments.aws_smoke import EMERGENCY_PROFILE_NAME
from experiments.aws_smoke.memory import EmergencyMemoryProbeEvidence, PrecisionMode

RunArm = Literal["oracle_sft", "ce_ablation", "logit_kd", "sequence_kd"]

REQUIRED_ARMS: tuple[RunArm, ...] = ("oracle_sft", "ce_ablation", "logit_kd")
OPTIONAL_ARMS: tuple[RunArm, ...] = ("sequence_kd",)
ALL_ARMS: tuple[RunArm, ...] = REQUIRED_ARMS + OPTIONAL_ARMS
DEFAULT_UNIQUE_LAUNCH_ORDER: tuple[RunArm, ...] = (
    "oracle_sft",
    "sequence_kd",
    "logit_kd",
)
CONTROL_LAUNCH_ORDER: tuple[RunArm, ...] = (
    "oracle_sft",
    "logit_kd",
    "ce_ablation",
)

INSTANCE_TYPE = "ml.g5.xlarge"
HOURLY_USD = 1.408
MAX_RUNTIME_SECONDS = 15 * 60
QUOTA_INSTANCE_COUNT = 1


class EmergencyTrainingProfile(BaseModel):
    """Sealed emergency knobs shared by all arms."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = EMERGENCY_PROFILE_NAME
    seed: StrictInt = 17
    train_examples: StrictInt = Field(default=32, ge=32, le=64)
    validation_examples: StrictInt = Field(default=16, ge=16, le=16)
    max_steps: StrictInt = Field(default=8, ge=5, le=10)
    max_length: StrictInt = 640
    max_completion: StrictInt = 128
    microbatch: StrictInt = 1
    grad_accumulation: StrictInt = 1
    gradient_checkpointing: bool = True
    learning_rate: float = Field(default=2e-4, gt=0.0)
    lora_rank: StrictInt = 8
    lora_alpha: StrictInt = 16
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    logit_temperature: float = Field(default=2.0, gt=0.0)
    kd_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    hard_ce_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    vocab_chunk: StrictInt = 4096
    precision_mode: PrecisionMode = "qlora_nf4"
    memory_probe_evidence: EmergencyMemoryProbeEvidence | None = None
    deterministic_algorithms: bool = True
    instance_type: str = INSTANCE_TYPE
    max_runtime_seconds: StrictInt = MAX_RUNTIME_SECONDS
    artifact_reserve_seconds: StrictInt = Field(default=180, ge=120, le=300)
    shutdown_margin_seconds: StrictInt = Field(default=30, ge=15, le=60)
    hourly_usd: float = HOURLY_USD
    quota_instance_count: StrictInt = QUOTA_INSTANCE_COUNT

    @model_validator(mode="after")
    def _validate_protocol(self) -> EmergencyTrainingProfile:
        if abs(self.kd_weight + self.hard_ce_weight - 1.0) > 1e-9:
            raise ValueError("kd_weight + hard_ce_weight must equal 1.0")
        if not self.gradient_checkpointing:
            raise ValueError("emergency profile requires gradient_checkpointing=true")
        if not self.deterministic_algorithms:
            raise ValueError("emergency profile requires deterministic_algorithms=true")
        if self.precision_mode == "bf16_lora" and self.memory_probe_evidence is None:
            raise ValueError("bf16_lora requires sealed emergency memory-probe evidence")
        if (
            self.memory_probe_evidence is not None
            and self.memory_probe_evidence.precision_mode != self.precision_mode
        ):
            raise ValueError("memory probe precision mode must match sealed profile")
        return self

    @property
    def max_run_usd(self) -> float:
        """Per-run ceiling: MaxRuntime * locked hourly price (rounded up to cents)."""
        gross = self.hourly_usd * (self.max_runtime_seconds / 3600.0)
        return math.ceil(gross * 100.0) / 100.0

    @property
    def estimate_high_usd(self) -> float:
        return self.max_run_usd

    @property
    def estimate_low_usd(self) -> float:
        # Honest lower bound: assume jobs often finish near the step budget (~6 min).
        low = self.hourly_usd * (6.0 / 60.0)
        return math.floor(low * 100.0) / 100.0


DEFAULT_EMERGENCY_PROFILE = EmergencyTrainingProfile()


def arm_recipe_resolved(arm: RunArm) -> str:
    if arm in {"oracle_sft", "sequence_kd"}:
        return "sequence.v1"
    if arm in {"ce_ablation", "logit_kd"}:
        return "logit.v1"
    raise ValueError(f"unknown arm: {arm}")


def arm_objective(
    arm: RunArm,
    profile: EmergencyTrainingProfile | None = None,
) -> dict[str, Any]:
    p = profile or DEFAULT_EMERGENCY_PROFILE
    if arm == "oracle_sft":
        return {
            "mode": "oracle_sft",
            "signal": "oracle_hard_ce",
            "kd_weight": 0.0,
            "hard_ce_weight": 1.0,
            "hard_target_source": "oracle",
            "teacher_runtime": "omitted",
            "scientific_role": "hard_target_baseline",
            "distinct_training_signal": True,
            "equivalent_to": None,
            "treatment_overhead": "student_only",
        }
    if arm == "sequence_kd":
        return {
            "mode": "sequence_kd",
            "signal": "teacher_sequence_ce",
            "kd_weight": 0.0,
            "hard_ce_weight": 1.0,
            "hard_target_source": "pre_materialized_teacher",
            "teacher_runtime": "offline_pre_materialized_only",
            "scientific_role": "distinct_sequence_distillation",
            "distinct_training_signal": True,
            "equivalent_to": None,
            "treatment_overhead": "teacher_materialization_excluded_from_job",
        }
    if arm == "ce_ablation":
        return {
            "mode": "ce_ablation",
            "signal": "oracle_hard_ce",
            "kd_weight": 0.0,
            "hard_ce_weight": 1.0,
            "hard_target_source": "oracle",
            "teacher_runtime": "omitted",
            "scientific_role": "matched_control_for_logit_kd",
            "distinct_training_signal": False,
            "equivalent_to": "oracle_sft",
            "treatment_overhead": "student_only",
            "equivalence_reason": (
                "identical oracle targets and hard-CE objective; this is a matched "
                "replication/control, not a distinct method"
            ),
        }
    if arm == "logit_kd":
        return {
            "mode": "logit_kd",
            "signal": "forward_kl_full_vocab",
            "kd_weight": p.kd_weight,
            "hard_ce_weight": p.hard_ce_weight,
            "hard_target_source": "oracle",
            "teacher_runtime": "online_frozen_forward_per_microbatch",
            "scientific_role": "distinct_logit_distillation",
            "distinct_training_signal": True,
            "equivalent_to": None,
            "treatment_overhead": "teacher_memory_and_forward_runtime_included",
        }
    raise ValueError(f"unknown arm: {arm}")


def default_launch_order(
    available_arms: set[RunArm],
    *,
    require_three_distinct: bool = True,
) -> tuple[RunArm, ...]:
    if set(DEFAULT_UNIQUE_LAUNCH_ORDER) <= available_arms:
        return DEFAULT_UNIQUE_LAUNCH_ORDER
    if require_three_distinct:
        raise ValueError(
            "three distinct default signals require sequence_kd with sealed "
            "pre-materialized teacher-response evidence; only oracle_sft and "
            "logit_kd are distinct without it"
        )
    if not set(CONTROL_LAUNCH_ORDER) <= available_arms:
        missing = sorted(set(CONTROL_LAUNCH_ORDER) - available_arms)
        raise ValueError(f"control launch order missing arms: {missing}")
    return CONTROL_LAUNCH_ORDER
