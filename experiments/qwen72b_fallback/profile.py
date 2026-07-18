"""Sealed Qwen2.5-72B QLoRA oracle/sequence-SFT training profiles."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from experiments.qwen72b_fallback import FALLBACK_ROLE_NAME, QWEN72B_PROFILE_NAME
from experiments.qwen72b_fallback.cost import (
    FULL_RUN_HARD_CAP_USD,
    P4DE_HOURLY_USD,
    P4DE_PRICE_SOURCE,
    REHEARSAL_HARD_CAP_USD,
    assert_under_cap,
    exact_gross_cost_usd,
)
from experiments.qwen72b_fallback.deadline import (
    ARTIFACT_RESERVE_SECONDS,
    FULL_MAX_RUNTIME_SECONDS,
    REHEARSAL_MAX_RUNTIME_SECONDS,
    SHUTDOWN_MARGIN_SECONDS,
)
from experiments.qwen72b_fallback.memory import (
    Qwen72BMemoryProbeEvidence,
    choose_precision_plan,
)
from experiments.qwen72b_fallback.pins import MODEL_ID, REVISION

INSTANCE_TYPE = "ml.p4de.24xlarge"
WORLD_SIZE = 8
SEQ_CAP = 1024
MICROBATCH = 1
GLOBAL_BATCH = 8
LORA_RANK = 16
MODEL_ROLE = FALLBACK_ROLE_NAME


class Qwen72BTrainingProfile(BaseModel):
    """Sealed knobs for oracle/sequence-SFT adaptation of the 72B base."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = QWEN72B_PROFILE_NAME
    kind: Literal["rehearsal", "full"]
    objective_mode: Literal["oracle_sequence_sft"] = "oracle_sequence_sft"
    model_role: Literal["oracle_sft_adapted_fallback"] = MODEL_ROLE
    seed: StrictInt = 17
    train_examples: StrictInt = Field(ge=1)
    max_updates: StrictInt = Field(ge=1)
    max_length: StrictInt = Field(default=SEQ_CAP, ge=SEQ_CAP, le=SEQ_CAP)
    microbatch: StrictInt = Field(default=MICROBATCH, ge=MICROBATCH, le=MICROBATCH)
    world_size: StrictInt = Field(default=WORLD_SIZE, ge=WORLD_SIZE, le=WORLD_SIZE)
    global_batch: StrictInt = Field(default=GLOBAL_BATCH, ge=GLOBAL_BATCH, le=GLOBAL_BATCH)
    grad_accumulation: StrictInt = Field(default=1, ge=1, le=1)
    gradient_checkpointing: bool = True
    learning_rate: float = Field(default=5e-5, gt=0.0)
    lora_rank: StrictInt = Field(default=LORA_RANK, ge=LORA_RANK, le=LORA_RANK)
    lora_alpha: StrictInt = Field(default=32, ge=32, le=32)
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
    precision_mode: Literal["qlora_4bit"] = "qlora_4bit"
    distributed_strategy: Literal["ddp"] = "ddp"
    flash_attention_2: bool = True
    packed_completion_only: bool = True
    deterministic_algorithms: bool = True
    instance_type: Literal["ml.p4de.24xlarge"] = INSTANCE_TYPE
    model_id: Literal["Qwen/Qwen2.5-72B-Instruct"] = MODEL_ID
    model_revision: str = REVISION
    # Trajectories / tool traces are precomputed outside the warm timer.
    trajectory_runtime: Literal["offline_precomputed_only"] = "offline_precomputed_only"
    data_policy: Literal["synthetic_finance_only"] = "synthetic_finance_only"
    max_runtime_seconds: StrictInt
    artifact_reserve_seconds: StrictInt = ARTIFACT_RESERVE_SECONDS
    shutdown_margin_seconds: StrictInt = SHUTDOWN_MARGIN_SECONDS
    hourly_usd: float = P4DE_HOURLY_USD
    price_source: str = P4DE_PRICE_SOURCE
    hard_cap_usd: float
    rehearsal_optimizer_steps: StrictInt | None = None
    memory_probe_evidence: Qwen72BMemoryProbeEvidence | None = None
    is_distilled_student: bool = False

    @model_validator(mode="after")
    def _validate_protocol(self) -> Qwen72BTrainingProfile:
        if self.is_distilled_student:
            raise ValueError("72B fallback profile must not claim distilled-student status")
        if self.grad_accumulation * self.microbatch * self.world_size != self.global_batch:
            raise ValueError(
                "global_batch must equal microbatch * world_size * grad_accumulation"
            )
        if self.train_examples != self.max_updates * self.global_batch:
            raise ValueError("train_examples must equal max_updates * global_batch")
        if not self.gradient_checkpointing:
            raise ValueError("72B profile requires gradient_checkpointing=true")
        if not self.deterministic_algorithms:
            raise ValueError("72B profile requires deterministic_algorithms=true")
        if not self.packed_completion_only:
            raise ValueError("72B profile requires packed completion-only sequences")
        if self.distributed_strategy != "ddp":
            raise ValueError(
                "default sealed strategy is DDP; FSDP2/ZeRO require a measured probe path"
            )
        if self.precision_mode != "qlora_4bit":
            raise ValueError("sealed profile chose qlora_4bit from the memory plan")
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
        if self.kind == "rehearsal":
            if self.max_runtime_seconds != REHEARSAL_MAX_RUNTIME_SECONDS:
                raise ValueError("rehearsal max_runtime_seconds mismatch")
            if self.hard_cap_usd != REHEARSAL_HARD_CAP_USD:
                raise ValueError("rehearsal hard cap must be $100")
            if self.rehearsal_optimizer_steps != 3:
                raise ValueError("rehearsal must seal exactly 3 optimizer steps")
            if self.max_updates != 3:
                raise ValueError("rehearsal max_updates must equal 3")
        else:
            if self.max_runtime_seconds != FULL_MAX_RUNTIME_SECONDS:
                raise ValueError("full run max_runtime_seconds must be 90 minutes")
            if self.hard_cap_usd != FULL_RUN_HARD_CAP_USD:
                raise ValueError("full run hard cap must be $500")
            if self.max_updates < 30:
                raise ValueError("full run must schedule a meaningful update budget")
        plan = choose_precision_plan(
            max_length=self.max_length,
            microbatch=self.microbatch,
            lora_rank=self.lora_rank,
            measured_probe=self.memory_probe_evidence,
        )
        if plan["chosen_precision_mode"] != self.precision_mode:
            raise ValueError("profile precision diverged from memory plan")
        if plan["chosen_distributed_strategy"] != self.distributed_strategy:
            raise ValueError("profile strategy diverged from memory plan")
        gross = exact_gross_cost_usd(
            hourly_usd=self.hourly_usd,
            max_runtime_seconds=self.max_runtime_seconds,
        )
        assert_under_cap(
            gross_usd=gross,
            hard_cap_usd=self.hard_cap_usd,
            label=f"qwen72b_{self.kind}",
        )
        return self

    @property
    def max_run_usd(self) -> float:
        gross = self.hourly_usd * (self.max_runtime_seconds / 3600.0)
        return math.ceil(gross * 100.0) / 100.0

    def objective_dict(self) -> dict[str, Any]:
        return {
            "mode": self.objective_mode,
            "signal": "synthetic_oracle_sequence_ce",
            "kd_weight": 0.0,
            "hard_ce_weight": 1.0,
            "hard_target_source": "synthetic_oracle",
            "trajectory_runtime": self.trajectory_runtime,
            "scientific_role": self.model_role,
            "is_distilled_student": False,
            "deployable_small_model": "TinyFable",
            "quality_fallback_model": "Qwen2.5-72B-Instruct oracle-SFT adapted",
            "distinct_training_signal": True,
            "treatment_overhead": "teacher_and_tool_trajectories_excluded_from_warm_timer",
        }

    def memory_plan(self) -> dict[str, Any]:
        return choose_precision_plan(
            max_length=self.max_length,
            microbatch=self.microbatch,
            lora_rank=self.lora_rank,
            measured_probe=self.memory_probe_evidence,
        )


def rehearsal_profile() -> Qwen72BTrainingProfile:
    return Qwen72BTrainingProfile(
        kind="rehearsal",
        train_examples=24,
        max_updates=3,
        max_runtime_seconds=REHEARSAL_MAX_RUNTIME_SECONDS,
        hard_cap_usd=REHEARSAL_HARD_CAP_USD,
        rehearsal_optimizer_steps=3,
    )


def full_profile() -> Qwen72BTrainingProfile:
    # 90-minute ceiling, one deterministic epoch-style slice: 240 updates.
    return Qwen72BTrainingProfile(
        kind="full",
        train_examples=1920,
        max_updates=240,
        max_runtime_seconds=FULL_MAX_RUNTIME_SECONDS,
        hard_cap_usd=FULL_RUN_HARD_CAP_USD,
        rehearsal_optimizer_steps=None,
    )
