"""Sealed huge-backup warm training profile (offline sequence distillation)."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from experiments.huge_backup import HUGE_BACKUP_PROFILE_NAME, OBJECTIVE_MODE
from experiments.huge_backup.deadline import (
    ARTIFACT_RESERVE_SECONDS,
    MAX_RUNTIME_SECONDS,
    SHUTDOWN_MARGIN_SECONDS,
)
from experiments.huge_backup.memory import HugeBackupMemoryProbeEvidence

LoraRank = Literal[16, 32]
INSTANCE_TYPE = "ml.p4de.24xlarge"
# Operator-attested SageMaker on-demand us-east-1 (Holori/CloudPrice 2026-06).
HOURLY_USD = 31.5641
PRICE_SOURCE = "operator_attested_ml.p4de.24xlarge_us-east-1_31.5641"
WORLD_SIZE = 8
TRAIN_EXAMPLES = 3200
MAX_UPDATES = 200
GLOBAL_BATCH = 16
MICROBATCH = 1
SEQ_CAP = 768
STUDENT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
TEACHER_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
FALLBACK_STUDENT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class HugeBackupTrainingProfile(BaseModel):
    """Sealed knobs for one deterministic epoch of offline sequence distillation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = HUGE_BACKUP_PROFILE_NAME
    objective_mode: Literal["offline_sequence_distillation"] = OBJECTIVE_MODE
    seed: StrictInt = 17
    # Production seal is 3200/200/16/8; tests may scale while preserving ratios.
    train_examples: StrictInt = Field(default=TRAIN_EXAMPLES, ge=16)
    max_updates: StrictInt = Field(default=MAX_UPDATES, ge=1)
    max_length: StrictInt = Field(default=SEQ_CAP, ge=8, le=SEQ_CAP)
    microbatch: StrictInt = Field(default=MICROBATCH, ge=1, le=1)
    world_size: StrictInt = Field(default=WORLD_SIZE, ge=1, le=WORLD_SIZE)
    global_batch: StrictInt = Field(default=GLOBAL_BATCH, ge=1)
    grad_accumulation: StrictInt = Field(default=2, ge=1)
    gradient_checkpointing: bool = True
    learning_rate: float = Field(default=1e-4, gt=0.0)
    lora_rank: LoraRank = 16
    lora_alpha: StrictInt = 32
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
    precision_mode: Literal["bf16_lora"] = "bf16_lora"
    distributed_strategy: Literal["ddp"] = "ddp"
    flash_attention_2: bool = True
    packed_completion_only: bool = True
    deterministic_algorithms: bool = True
    instance_type: Literal["ml.p4de.24xlarge"] = INSTANCE_TYPE
    student_model_id: Literal["Qwen/Qwen2.5-14B-Instruct"] = STUDENT_MODEL_ID
    teacher_model_id: Literal["Qwen/Qwen2.5-32B-Instruct"] = TEACHER_MODEL_ID
    teacher_runtime: Literal["offline_pre_materialized_only"] = "offline_pre_materialized_only"
    max_runtime_seconds: StrictInt = MAX_RUNTIME_SECONDS
    artifact_reserve_seconds: StrictInt = ARTIFACT_RESERVE_SECONDS
    shutdown_margin_seconds: StrictInt = SHUTDOWN_MARGIN_SECONDS
    hourly_usd: float = HOURLY_USD
    price_source: str = PRICE_SOURCE
    rehearsal_optimizer_steps: StrictInt = Field(default=3, ge=3, le=3)
    rehearsal_median_step_fail_seconds: float = Field(default=8.0, gt=0.0)
    memory_probe_evidence: HugeBackupMemoryProbeEvidence | None = None

    @model_validator(mode="after")
    def _validate_protocol(self) -> HugeBackupTrainingProfile:
        if self.grad_accumulation * self.microbatch * self.world_size != self.global_batch:
            raise ValueError(
                "global_batch must equal microbatch * world_size * grad_accumulation "
                f"({self.microbatch}*{self.world_size}*{self.grad_accumulation})"
            )
        if self.train_examples != self.max_updates * self.global_batch:
            raise ValueError(
                "train_examples must equal max_updates * global_batch for one "
                f"deterministic epoch ({self.max_updates}*{self.global_batch})"
            )
        if not self.gradient_checkpointing:
            raise ValueError("huge_backup requires gradient_checkpointing=true")
        if not self.deterministic_algorithms:
            raise ValueError("huge_backup requires deterministic_algorithms=true")
        if not self.packed_completion_only:
            raise ValueError("huge_backup requires packed completion-only sequences")
        if self.distributed_strategy != "ddp":
            raise ValueError("huge_backup seals DDP; FSDP/ZeRO require a measured probe")
        if self.lora_rank not in (16, 32):
            raise ValueError("lora_rank must be 16 or 32")
        if self.lora_alpha != self.lora_rank * 2:
            raise ValueError("lora_alpha must be 2 * lora_rank")
        expected_targets = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        if set(self.lora_target_modules) != expected_targets:
            raise ValueError("LoRA targets must cover attention+MLP projections exactly")
        if self.memory_probe_evidence is not None:
            if self.memory_probe_evidence.precision_mode != self.precision_mode:
                raise ValueError("memory probe precision mode must match sealed profile")
            if self.memory_probe_evidence.instance_type != self.instance_type:
                raise ValueError("memory probe instance_type must be ml.p4de.24xlarge")
        return self

    @property
    def max_run_usd(self) -> float:
        """Per-run ceiling: MaxRuntime * locked hourly price (rounded up to cents)."""
        gross = self.hourly_usd * (self.max_runtime_seconds / 3600.0)
        return math.ceil(gross * 100.0) / 100.0

    def objective_dict(self) -> dict[str, Any]:
        return {
            "mode": self.objective_mode,
            "signal": "teacher_sequence_ce",
            "kd_weight": 0.0,
            "hard_ce_weight": 1.0,
            "hard_target_source": "pre_materialized_teacher",
            "teacher_runtime": self.teacher_runtime,
            "scientific_role": "offline_sequence_distillation",
            "distinct_training_signal": True,
            "equivalent_to": None,
            "treatment_overhead": "teacher_materialization_excluded_from_warm_timer",
            "not_exact_logit_kd": True,
        }


DEFAULT_HUGE_BACKUP_PROFILE = HugeBackupTrainingProfile()


def assert_production_seal(profile: HugeBackupTrainingProfile) -> None:
    """Fail closed unless the profile matches the frozen warm-job seal."""
    expected = {
        "train_examples": TRAIN_EXAMPLES,
        "max_updates": MAX_UPDATES,
        "global_batch": GLOBAL_BATCH,
        "microbatch": MICROBATCH,
        "world_size": WORLD_SIZE,
        "max_length": SEQ_CAP,
        "instance_type": INSTANCE_TYPE,
        "student_model_id": STUDENT_MODEL_ID,
        "teacher_model_id": TEACHER_MODEL_ID,
        "distributed_strategy": "ddp",
        "precision_mode": "bf16_lora",
        "objective_mode": OBJECTIVE_MODE,
    }
    actual = {key: getattr(profile, key) for key in expected}
    if actual != expected:
        raise ValueError(f"production seal mismatch: expected={expected} actual={actual}")
    if profile.max_runtime_seconds != 1800 or profile.artifact_reserve_seconds != 300:
        raise ValueError("production seal requires 1800s runtime with 300s artifact reserve")
