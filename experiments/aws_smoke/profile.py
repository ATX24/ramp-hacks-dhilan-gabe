"""Fixed emergency training profile: 5–10 steps, tiny corpus, 15-minute ceiling."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from experiments.aws_smoke import EMERGENCY_PROFILE_NAME

RunArm = Literal["oracle_sft", "ce_ablation", "logit_kd", "sequence_kd"]

REQUIRED_ARMS: tuple[RunArm, ...] = ("oracle_sft", "ce_ablation", "logit_kd")
OPTIONAL_ARMS: tuple[RunArm, ...] = ("sequence_kd",)

INSTANCE_TYPE = "ml.g5.xlarge"
HOURLY_USD = 1.408
MAX_RUNTIME_SECONDS = 15 * 60
QUOTA_INSTANCE_COUNT = 1


class EmergencyTrainingProfile(BaseModel):
    """Sealed emergency knobs shared by all arms."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = EMERGENCY_PROFILE_NAME
    seed: StrictInt = 17
    train_examples: StrictInt = Field(default=48, ge=32, le=64)
    validation_examples: StrictInt = Field(default=16, ge=16, le=16)
    max_steps: StrictInt = Field(default=8, ge=5, le=10)
    max_length: StrictInt = 512
    max_completion: StrictInt = 128
    microbatch: StrictInt = 1
    grad_accumulation: StrictInt = 1
    learning_rate: float = Field(default=2e-4, gt=0.0)
    lora_rank: StrictInt = 8
    lora_alpha: StrictInt = 16
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    logit_temperature: float = Field(default=2.0, gt=0.0)
    kd_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    hard_ce_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    vocab_chunk: StrictInt = 4096
    instance_type: str = INSTANCE_TYPE
    max_runtime_seconds: StrictInt = MAX_RUNTIME_SECONDS
    hourly_usd: float = HOURLY_USD
    quota_instance_count: StrictInt = QUOTA_INSTANCE_COUNT

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


def arm_objective(arm: RunArm) -> dict[str, float | str]:
    if arm == "oracle_sft":
        return {
            "mode": "oracle_sft",
            "signal": "oracle_hard_ce",
            "kd_weight": 0.0,
            "hard_ce_weight": 1.0,
        }
    if arm == "sequence_kd":
        return {
            "mode": "sequence_kd",
            "signal": "teacher_sequence_ce",
            "kd_weight": 0.0,
            "hard_ce_weight": 1.0,
        }
    if arm == "ce_ablation":
        return {
            "mode": "ce_ablation",
            "signal": "hard_ce_only",
            "kd_weight": 0.0,
            "hard_ce_weight": 1.0,
        }
    if arm == "logit_kd":
        return {
            "mode": "logit_kd",
            "signal": "forward_kl_full_vocab",
            "kd_weight": 0.7,
            "hard_ce_weight": 0.3,
        }
    raise ValueError(f"unknown arm: {arm}")
