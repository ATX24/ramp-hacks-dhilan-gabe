"""Task envelopes and typed outputs for the synthetic finance world."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    Field,
    JsonValue,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import (
    NonNegativeSafeInt,
    PositiveSafeInt,
    SafeInt,
)
from distillery.contracts.ids import ExampleId, GroupId, WorldId

SCHEMA_VERSION_FINANCE_WORLD = "finance_world.v1"
SCHEMA_VERSION_FINANCE_WORLD_V2 = "finance_world.v2"
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]

# Canonical spend categories for merchant_tagging (bounded closed set).
MERCHANT_SPEND_CATEGORIES: frozenset[str] = frozenset(
    {
        "meals",
        "airfare",
        "lodging",
        "saas",
        "cloud",
        "capex",
        "services",
        "fees",
        "facilities",
        "personal",
        "rideshare",
        "ground_transport",
        "office_supplies",
        "advertising",
    }
)

# Bounded finance tags for merchant_tagging outputs.
MERCHANT_FINANCE_TAGS: frozenset[str] = frozenset(
    {
        "recurring",
        "travel",
        "entertainment",
        "infrastructure",
        "software",
        "hardware",
        "professional",
        "processor",
        "employee_spend",
        "vendor_bill",
        "card_present",
        "card_not_present",
        "international",
        "refundable",
    }
)


class TaskId(StrEnum):
    TRANSACTION_REVIEW = "transaction_review"
    VARIANCE_ANALYSIS = "variance_analysis"
    CASH_RECONCILIATION = "cash_reconciliation"
    MERCHANT_TAGGING = "merchant_tagging"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    IID_TEST = "iid_test"
    OOD_TEST = "ood_test"


class LabelSource(StrEnum):
    ORACLE = "oracle"
    TEACHER = "teacher"
    IMPORTED = "imported"
    RULES = "rules"


class EvidenceRef(FrozenModel):
    source_id: StrictStr = Field(min_length=1)
    field: StrictStr = Field(min_length=1)
    value: StrictStr


class JournalLine(FrozenModel):
    account: StrictStr = Field(min_length=1)
    side: Literal["debit", "credit"]
    amount_minor: NonNegativeSafeInt


class TransactionReviewOutput(FrozenModel):
    schema_version: Literal["transaction_review.v1"] = "transaction_review.v1"
    task: Literal[TaskId.TRANSACTION_REVIEW] = TaskId.TRANSACTION_REVIEW
    gl_account: StrictStr = Field(min_length=1)
    journal_entry: tuple[JournalLine, ...] = Field(min_length=2)
    policy_action: Literal["approve", "review", "reject"]
    rule_ids: tuple[StrictStr, ...]
    evidence: tuple[EvidenceRef, ...]
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _balanced_journal(self) -> TransactionReviewOutput:
        debits = sum(line.amount_minor for line in self.journal_entry if line.side == "debit")
        credits = sum(line.amount_minor for line in self.journal_entry if line.side == "credit")
        if not any(
            line.side == "debit" and line.amount_minor > 0
            for line in self.journal_entry
        ):
            raise ValueError("journal entry requires at least one positive debit")
        if not any(
            line.side == "credit" and line.amount_minor > 0
            for line in self.journal_entry
        ):
            raise ValueError("journal entry requires at least one positive credit")
        if debits != credits:
            raise ValueError(
                f"journal entry unbalanced: debits={debits} credits={credits}"
            )
        return self


class VarianceDriver(FrozenModel):
    driver_id: StrictStr = Field(min_length=1)
    impact_minor: SafeInt
    rank: PositiveSafeInt


class VarianceAnalysisOutput(FrozenModel):
    schema_version: Literal["variance_analysis.v1"] = "variance_analysis.v1"
    task: Literal[TaskId.VARIANCE_ANALYSIS] = TaskId.VARIANCE_ANALYSIS
    profit_impact_minor: SafeInt
    direction: Literal["favorable", "unfavorable"]
    top_drivers: tuple[VarianceDriver, ...] = Field(min_length=1)
    other_impact_minor: SafeInt
    rule_ids: tuple[StrictStr, ...]
    evidence_ids: tuple[StrictStr, ...]
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _arithmetic_closure(self) -> VarianceAnalysisOutput:
        if self.profit_impact_minor == 0:
            raise ValueError(
                "profit_impact_minor cannot be zero when direction is "
                "favorable/unfavorable"
            )
        expected_ranks = list(range(1, len(self.top_drivers) + 1))
        actual_ranks = [driver.rank for driver in self.top_drivers]
        if actual_ranks != expected_ranks:
            raise ValueError(
                "variance driver ranks must be unique, contiguous, and ordered from 1"
            )
        driver_ids = [driver.driver_id for driver in self.top_drivers]
        if len(driver_ids) != len(set(driver_ids)):
            raise ValueError("variance driver_id values must be unique")
        for previous, current in zip(self.top_drivers, self.top_drivers[1:], strict=False):
            previous_abs = abs(previous.impact_minor)
            current_abs = abs(current.impact_minor)
            if previous_abs < current_abs:
                raise ValueError(
                    "variance drivers must be ranked by descending absolute impact"
                )
            if previous_abs == current_abs and previous.driver_id >= current.driver_id:
                raise ValueError(
                    "equal-impact variance drivers must use ascending driver_id tie-break"
                )
        driver_sum = sum(d.impact_minor for d in self.top_drivers)
        if driver_sum + self.other_impact_minor != self.profit_impact_minor:
            raise ValueError(
                "variance arithmetic not closed: "
                f"drivers+other={driver_sum + self.other_impact_minor} "
                f"!= profit_impact={self.profit_impact_minor}"
            )
        expected_dir: Literal["favorable", "unfavorable"] = (
            "favorable" if self.profit_impact_minor > 0 else "unfavorable"
        )
        if self.direction != expected_dir:
            raise ValueError(
                f"direction {self.direction!r} inconsistent with "
                f"profit_impact_minor={self.profit_impact_minor}"
            )
        return self


class MatchedGroup(FrozenModel):
    book_ids: tuple[StrictStr, ...] = Field(min_length=1)
    bank_ids: tuple[StrictStr, ...] = Field(min_length=1)


class ReconciliationException(FrozenModel):
    type: Literal[
        "bank_fee",
        "deposit_in_transit",
        "stale_check",
        "duplicate",
        "unexplained",
        "partial_settlement",
    ]
    event_ids: tuple[StrictStr, ...] = Field(min_length=1)
    amount_minor: SafeInt


class CashReconciliationOutput(FrozenModel):
    schema_version: Literal["cash_reconciliation.v1"] = "cash_reconciliation.v1"
    task: Literal[TaskId.CASH_RECONCILIATION] = TaskId.CASH_RECONCILIATION
    status: Literal["balanced", "exceptions"]
    matched_groups: tuple[MatchedGroup, ...]
    exceptions: tuple[ReconciliationException, ...]
    adjusted_book_balance_minor: SafeInt
    adjusted_bank_balance_minor: SafeInt
    difference_minor: SafeInt
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _difference_matches(self) -> CashReconciliationOutput:
        expected = self.adjusted_book_balance_minor - self.adjusted_bank_balance_minor
        if self.difference_minor != expected:
            raise ValueError(
                f"difference_minor={self.difference_minor} != "
                f"book-bank={expected}"
            )
        if self.status == "balanced":
            if self.exceptions:
                raise ValueError("balanced reconciliation cannot contain exceptions")
            if self.difference_minor != 0:
                raise ValueError("balanced reconciliation must have zero difference")
        elif not self.exceptions:
            raise ValueError("exceptions reconciliation requires at least one exception")
        return self


class MerchantTaggingOutput(FrozenModel):
    """Compact merchant identity + category/tag contract (Primary C)."""

    schema_version: Literal["merchant_tagging.v1"] = "merchant_tagging.v1"
    task: Literal[TaskId.MERCHANT_TAGGING] = TaskId.MERCHANT_TAGGING
    merchant_id: StrictStr = Field(min_length=1)
    merchant_name: StrictStr = Field(min_length=1)
    spend_category: StrictStr = Field(min_length=1)
    tags: tuple[StrictStr, ...] = Field(min_length=1, max_length=6)
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _bounded_category_and_tags(self) -> MerchantTaggingOutput:
        if self.spend_category not in MERCHANT_SPEND_CATEGORIES:
            raise ValueError(
                f"spend_category {self.spend_category!r} not in closed category set"
            )
        unknown = sorted(set(self.tags) - MERCHANT_FINANCE_TAGS)
        if unknown:
            raise ValueError(f"unknown finance tags: {unknown}")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("merchant tags must be unique")
        if list(self.tags) != sorted(self.tags):
            raise ValueError("merchant tags must be sorted ascending")
        return self


TaskOutput = Annotated[
    TransactionReviewOutput
    | VarianceAnalysisOutput
    | CashReconciliationOutput
    | MerchantTaggingOutput,
    Field(discriminator="task"),
]
_TASK_OUTPUT_ADAPTER = TypeAdapter(TaskOutput)


def validate_task_output(value: Mapping[str, JsonValue]) -> TaskOutput:
    """Validate and return one of the executable task output contracts."""
    return _TASK_OUTPUT_ADAPTER.validate_python(value)


class OracleMeta(FrozenModel):
    generator_revision: StrictStr = Field(min_length=1)
    latent_state_hash: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class Provenance(FrozenModel):
    split: SplitName
    template_family: StrictStr = Field(min_length=1)
    label_source: LabelSource


class FinanceTaskEnvelope(FrozenModel):
    """Canonical task envelope for finance_world.v1 / finance_world.v2 examples."""

    schema_version: Literal["finance_world.v1", "finance_world.v2"] = (
        SCHEMA_VERSION_FINANCE_WORLD
    )
    example_id: ExampleId
    world_id: WorldId
    group_id: GroupId
    task: TaskId
    difficulty: Difficulty
    input: FrozenJsonObject
    expected_output: FrozenJsonObject
    oracle: OracleMeta
    provenance: Provenance
    # Fixture-only markers for negative / edge cases (ignored by trainers).
    case_tags: tuple[StrictStr, ...] = ()

    @field_validator("expected_output")
    @classmethod
    def _non_empty_output(
        cls,
        value: Mapping[StrictStr, JsonValue],
    ) -> Mapping[StrictStr, JsonValue]:
        if not value:
            raise ValueError("expected_output must not be empty")
        return value

    @model_validator(mode="after")
    def _task_matches_strict_output(self) -> FinanceTaskEnvelope:
        output = validate_task_output(self.expected_output)
        if output.task != self.task:
            raise ValueError(
                f"envelope task {self.task.value!r} does not match "
                f"expected_output task {output.task.value!r}"
            )
        return self
