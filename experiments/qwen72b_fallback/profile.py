"""Sealed, non-authorizing 72B QLoRA profiles for probe/rehearsal/full runs."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distillery.contracts.hashing import content_sha256
from experiments.qwen72b_fallback.cost import (
    FULL_RUN_HARD_CAP_USD,
    P4DE_HOURLY_USD,
    P4DE_PRICE_SOURCE,
    PROBE_HARD_CAP_USD,
    REHEARSAL_HARD_CAP_USD,
    assert_under_cap,
    exact_gross_cost_usd,
)
from experiments.qwen72b_fallback.deadline import phase_budget_for
from experiments.qwen72b_fallback.memory import (
    AttentionBackend,
    DistributedStrategy,
    PrecisionMode,
    planning_comparison,
)
from experiments.qwen72b_fallback.pins import MODEL_ID, REVISION
from experiments.qwen72b_fallback.roles import (
    ModelRole,
    SupervisionKind,
    TrainingArm,
)

INSTANCE_TYPE = "ml.p4de.24xlarge"
WORLD_SIZE = 8
SEQ_CAP = 1024
MICROBATCH = 1
GLOBAL_BATCH = 8
LORA_RANK = 16


class RunKind(StrEnum):
    MEMORY_PROBE = "memory_probe"
    REHEARSAL = "rehearsal"
    FULL = "full"


class DeterminismScope(StrEnum):
    ORDER_SHAPES_AND_TORCH_OPS = "order_shapes_and_torch_deterministic_ops"


class Qwen72BTrainingProfile(BaseModel):
    """Exact profile. A separate measured probe authorizes execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["distillery.qwen72b_fallback.profile.v2"] = (
        "distillery.qwen72b_fallback.profile.v2"
    )
    name: Literal["qwen72b_finance_world_v2_fallback_v2"] = "qwen72b_finance_world_v2_fallback_v2"
    kind: RunKind
    model_role: Literal[ModelRole.QWEN72B_ADAPTED_FALLBACK] = ModelRole.QWEN72B_ADAPTED_FALLBACK
    training_arm: Literal[TrainingArm.FINANCE_WORLD_V2_ORACLE_SFT] = (
        TrainingArm.FINANCE_WORLD_V2_ORACLE_SFT
    )
    supervision_kind: Literal[SupervisionKind.FINANCE_WORLD_V2_LATENT_ORACLE] = (
        SupervisionKind.FINANCE_WORLD_V2_LATENT_ORACLE
    )
    model_id: Literal["Qwen/Qwen2.5-72B-Instruct"] = MODEL_ID
    model_revision: Literal["495f39366efef23836d0cfae4fbe635880d2be31"] = REVISION
    seed: Literal[17] = 17
    train_examples: int = Field(ge=8)
    max_updates: int = Field(ge=1)
    max_length: Literal[1024] = SEQ_CAP
    microbatch: Literal[1] = MICROBATCH
    world_size: Literal[8] = WORLD_SIZE
    global_batch: Literal[8] = GLOBAL_BATCH
    grad_accumulation: Literal[1] = 1
    gradient_checkpointing: Literal[True] = True
    learning_rate: Literal[5e-5] = 5e-5
    lora_rank: Literal[16] = LORA_RANK
    lora_alpha: Literal[32] = 32
    lora_dropout: Literal[0.05] = 0.05
    lora_target_modules: tuple[
        Literal["q_proj"],
        Literal["k_proj"],
        Literal["v_proj"],
        Literal["o_proj"],
        Literal["gate_proj"],
        Literal["up_proj"],
        Literal["down_proj"],
    ] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    precision_mode: Literal[PrecisionMode.QLORA_NF4_BF16] = PrecisionMode.QLORA_NF4_BF16
    distributed_strategy: Literal[DistributedStrategy.DDP] = DistributedStrategy.DDP
    attention_backend: Literal[AttentionBackend.SDPA_MATH] = AttentionBackend.SDPA_MATH
    flash_attention_2: Literal[False] = False
    packed_completion_only: Literal[True] = True
    fixed_shape_collation: Literal[True] = True
    deterministic_sampler: Literal[True] = True
    deterministic_algorithms: Literal[True] = True
    determinism_scope: Literal[DeterminismScope.ORDER_SHAPES_AND_TORCH_OPS] = (
        DeterminismScope.ORDER_SHAPES_AND_TORCH_OPS
    )
    bitwise_reproducibility_claimed: Literal[False] = False
    instance_type: Literal["ml.p4de.24xlarge"] = INSTANCE_TYPE
    data_world: Literal["finance_world.v2"] = "finance_world.v2"
    data_policy: Literal["synthetic_only"] = "synthetic_only"
    max_runtime_seconds: int
    hourly_usd: Literal[31.5641] = P4DE_HOURLY_USD
    price_source: Literal["operator_attested_ml.p4de.24xlarge_us-east-1_31.5641"] = (
        P4DE_PRICE_SOURCE
    )
    hard_cap_usd: float
    nccl_timeout_seconds: Literal[120] = 120

    @model_validator(mode="after")
    def _validate_profile(self) -> Qwen72BTrainingProfile:
        if self.grad_accumulation * self.microbatch * self.world_size != self.global_batch:
            raise ValueError("global batch arithmetic mismatch")
        if self.train_examples != self.max_updates * self.global_batch:
            raise ValueError("train_examples must equal max_updates * global_batch")
        if self.lora_alpha != self.lora_rank * 2:
            raise ValueError("lora_alpha must be twice lora_rank")
        budget = phase_budget_for(self.kind.value)
        if self.max_runtime_seconds != budget.max_runtime_seconds:
            raise ValueError("profile runtime differs from its explicit phase budget")
        expected_cap = {
            RunKind.MEMORY_PROBE: PROBE_HARD_CAP_USD,
            RunKind.REHEARSAL: REHEARSAL_HARD_CAP_USD,
            RunKind.FULL: FULL_RUN_HARD_CAP_USD,
        }[self.kind]
        if self.hard_cap_usd != expected_cap:
            raise ValueError("profile hard cap differs from run-kind policy")
        if self.kind in {RunKind.MEMORY_PROBE, RunKind.REHEARSAL}:
            if self.max_updates != 3:
                raise ValueError("probe/rehearsal must use exactly three optimizer steps")
        elif self.max_updates != 240:
            raise ValueError("full profile must use exactly 240 optimizer steps")
        gross = exact_gross_cost_usd(
            hourly_usd=self.hourly_usd,
            max_runtime_seconds=self.max_runtime_seconds,
        )
        assert_under_cap(
            gross_usd=gross,
            hard_cap_usd=self.hard_cap_usd,
            label=f"qwen72b_{self.kind.value}",
        )
        return self

    @property
    def profile_sha256(self) -> str:
        return content_sha256(self.model_dump(mode="json"))

    @property
    def max_run_usd(self) -> float:
        gross = self.hourly_usd * (self.max_runtime_seconds / 3600.0)
        return math.ceil(gross * 100.0) / 100.0

    def objective_dict(self) -> dict[str, Any]:
        return {
            "training_arm": self.training_arm.value,
            "supervision_kind": self.supervision_kind.value,
            "model_role": self.model_role.value,
            "data_world": self.data_world,
            "larger_teacher_model": None,
            "distilled_student": False,
        }

    def planning_memory_comparison(self) -> dict[str, object]:
        return planning_comparison()


def probe_profile() -> Qwen72BTrainingProfile:
    budget = phase_budget_for(RunKind.MEMORY_PROBE.value)
    return Qwen72BTrainingProfile(
        kind=RunKind.MEMORY_PROBE,
        train_examples=24,
        max_updates=3,
        max_runtime_seconds=budget.max_runtime_seconds,
        hard_cap_usd=PROBE_HARD_CAP_USD,
    )


def rehearsal_profile() -> Qwen72BTrainingProfile:
    budget = phase_budget_for(RunKind.REHEARSAL.value)
    return Qwen72BTrainingProfile(
        kind=RunKind.REHEARSAL,
        train_examples=24,
        max_updates=3,
        max_runtime_seconds=budget.max_runtime_seconds,
        hard_cap_usd=REHEARSAL_HARD_CAP_USD,
    )


def full_profile() -> Qwen72BTrainingProfile:
    budget = phase_budget_for(RunKind.FULL.value)
    return Qwen72BTrainingProfile(
        kind=RunKind.FULL,
        train_examples=1920,
        max_updates=240,
        max_runtime_seconds=budget.max_runtime_seconds,
        hard_cap_usd=FULL_RUN_HARD_CAP_USD,
    )
