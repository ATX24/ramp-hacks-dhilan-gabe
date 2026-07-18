"""Primary metrics, proof gates, and train/evaluation budget definitions."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from distillery.contracts.base import FrozenModel
from distillery.contracts.hashing import NonNegativeSafeInt, PositiveSafeInt

WEIGHT_SUM_TOLERANCE = 1e-9
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class PrimaryIndexWeights(FrozenModel):
    """Prespecified primary quality index weights (locked)."""

    transaction_joint_exact: FiniteFloat = Field(default=0.45, ge=0.0, le=1.0)
    variance_joint_exact: FiniteFloat = Field(default=0.45, ge=0.0, le=1.0)
    json_schema_validity: FiniteFloat = Field(default=0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> PrimaryIndexWeights:
        total = (
            self.transaction_joint_exact
            + self.variance_joint_exact
            + self.json_schema_validity
        )
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"primary index weights must sum to 1.0, got {total}")
        return self


PRIMARY_INDEX_WEIGHTS = PrimaryIndexWeights()


class ProofGates(FrozenModel):
    """Stop/go proof gates (locked constants for contracts-v1)."""

    teacher_gap_min_abs: FiniteFloat = Field(default=0.05, gt=0.0)
    quality_retention_point: FiniteFloat = Field(default=0.95, ge=0.0, le=1.0)
    quality_retention_lower_95: FiniteFloat = Field(default=0.90, ge=0.0, le=1.0)
    max_primary_task_regression: FiniteFloat = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )
    json_schema_validity_min: FiniteFloat = Field(default=0.99, ge=0.0, le=1.0)
    ood_retention_min: FiniteFloat = Field(default=0.90, ge=0.0, le=1.0)
    economics_utilization: FiniteFloat = Field(default=0.25, gt=0.0, le=1.0)
    required_seed_screen: NonNegativeSafeInt = 17
    required_seed_replication: NonNegativeSafeInt = 23
    preferred_seed_extra: NonNegativeSafeInt = 41
    pilot_examples: PositiveSafeInt = 200
    bootstrap_resamples: PositiveSafeInt = 10_000


class SmokeTrainingBudget(FrozenModel):
    train_examples: PositiveSafeInt = 320
    validation_examples: PositiveSafeInt = 80
    test_examples: PositiveSafeInt = 160
    max_length: PositiveSafeInt = 512
    max_completion: PositiveSafeInt = 160
    max_steps: PositiveSafeInt = 30
    lora_rank: PositiveSafeInt = 8
    lora_alpha: PositiveSafeInt = 16
    lora_dropout: FiniteFloat = Field(default=0.05, ge=0.0, le=1.0)
    microbatch: PositiveSafeInt = 1
    grad_accumulation: PositiveSafeInt = 8
    learning_rate: FiniteFloat = Field(default=2e-4, gt=0.0)
    seed: NonNegativeSafeInt = 17
    logit_temperature: FiniteFloat = Field(default=2.0, gt=0.0)
    kd_weight: FiniteFloat = Field(default=0.7, ge=0.0, le=1.0)
    hard_ce_weight: FiniteFloat = Field(default=0.3, ge=0.0, le=1.0)
    vocab_chunk: PositiveSafeInt = 4096
    max_runtime_seconds: PositiveSafeInt = 45 * 60

    @model_validator(mode="after")
    def _objective_weights_sum_to_one(self) -> SmokeTrainingBudget:
        total = self.kd_weight + self.hard_ce_weight
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"kd_weight + hard_ce_weight must equal 1.0, got {total}")
        return self


class TrainingBudget(FrozenModel):
    train_examples: PositiveSafeInt = 3200
    validation_examples: PositiveSafeInt = 400
    iid_test_examples: PositiveSafeInt = 800
    ood_test_examples: PositiveSafeInt = 800
    max_length: PositiveSafeInt = 768
    max_completion: PositiveSafeInt = 256
    max_steps: PositiveSafeInt = 200
    lora_rank: PositiveSafeInt = 16
    lora_alpha: PositiveSafeInt = 32
    lora_dropout: FiniteFloat = Field(default=0.05, ge=0.0, le=1.0)
    microbatch: PositiveSafeInt = 1
    grad_accumulation: PositiveSafeInt = 16
    learning_rate: FiniteFloat = Field(default=1e-4, gt=0.0)
    warmup_ratio: FiniteFloat = Field(default=0.05, ge=0.0, le=1.0)
    seed_screen: NonNegativeSafeInt = 17
    seed_replication: NonNegativeSafeInt = 23
    logit_temperature: FiniteFloat = Field(default=2.0, gt=0.0)
    kd_weight: FiniteFloat = Field(default=0.7, ge=0.0, le=1.0)
    hard_ce_weight: FiniteFloat = Field(default=0.3, ge=0.0, le=1.0)
    vocab_chunk: PositiveSafeInt = 4096
    max_runtime_seconds: PositiveSafeInt = 3 * 60 * 60
    default_max_run_usd: FiniteFloat = Field(default=25.0, gt=0.0)
    default_max_experiment_usd: FiniteFloat = Field(default=250.0, gt=0.0)

    @model_validator(mode="after")
    def _objective_weights_sum_to_one(self) -> TrainingBudget:
        total = self.kd_weight + self.hard_ce_weight
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"kd_weight + hard_ce_weight must equal 1.0, got {total}")
        return self


class EvaluationBudget(FrozenModel):
    warmup_requests: NonNegativeSafeInt = 20
    timed_examples_per_primary_task: PositiveSafeInt = 200
    batch_sizes: tuple[PositiveSafeInt, ...] = Field(default=(1, 8), min_length=1)
    mixture_transaction_review: FiniteFloat = Field(default=0.45, ge=0.0, le=1.0)
    mixture_variance_analysis: FiniteFloat = Field(default=0.45, ge=0.0, le=1.0)
    mixture_cash_reconciliation: FiniteFloat = Field(default=0.10, ge=0.0, le=1.0)
    difficulty_easy: FiniteFloat = Field(default=0.30, ge=0.0, le=1.0)
    difficulty_medium: FiniteFloat = Field(default=0.40, ge=0.0, le=1.0)
    difficulty_hard: FiniteFloat = Field(default=0.30, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _mixtures_sum_to_one(self) -> EvaluationBudget:
        task_total = (
            self.mixture_transaction_review
            + self.mixture_variance_analysis
            + self.mixture_cash_reconciliation
        )
        difficulty_total = (
            self.difficulty_easy + self.difficulty_medium + self.difficulty_hard
        )
        if abs(task_total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"task mixture weights must sum to 1.0, got {task_total}")
        if abs(difficulty_total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"difficulty mixture weights must sum to 1.0, got {difficulty_total}"
            )
        return self
