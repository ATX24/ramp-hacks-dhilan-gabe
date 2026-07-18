"""Immutable Dataset resource contract."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    Field,
    StrictStr,
    model_validator,
)

from distillery.contracts.base import FrozenDict, FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import (
    AwareDatetime,
    NonNegativeSafeInt,
    Sha256Hex,
    content_sha256,
)
from distillery.contracts.ids import DatasetId
from distillery.contracts.tasks import Difficulty, TaskId

FinanceWorldVersion = Literal["finance_world.v1", "finance_world.v2"]
FINANCE_WORLD_V1_TASK_IDS: frozenset[TaskId] = frozenset(
    {
        TaskId.TRANSACTION_REVIEW,
        TaskId.VARIANCE_ANALYSIS,
        TaskId.CASH_RECONCILIATION,
    }
)
FINANCE_WORLD_V2_TASK_IDS: frozenset[TaskId] = frozenset(
    {*FINANCE_WORLD_V1_TASK_IDS, TaskId.MERCHANT_TAGGING}
)
_SUPPORTED_TASK_SETS = (FINANCE_WORLD_V1_TASK_IDS, FINANCE_WORLD_V2_TASK_IDS)
_TASK_IDS_BY_FINANCE_WORLD: dict[FinanceWorldVersion, frozenset[TaskId]] = {
    "finance_world.v1": FINANCE_WORLD_V1_TASK_IDS,
    "finance_world.v2": FINANCE_WORLD_V2_TASK_IDS,
}


class SplitHashes(FrozenModel):
    train: Sha256Hex
    validation: Sha256Hex
    test: Sha256Hex | None = None
    iid_test: Sha256Hex | None = None
    ood_test: Sha256Hex | None = None


class TaskDifficultyCounts(FrozenModel):
    by_task: FrozenDict[TaskId, NonNegativeSafeInt]
    by_difficulty: FrozenDict[Difficulty, NonNegativeSafeInt]

    @model_validator(mode="after")
    def _complete_consistent_totals(self) -> TaskDifficultyCounts:
        if set(self.by_task) not in _SUPPORTED_TASK_SETS:
            raise ValueError(
                "by_task must contain exactly one complete supported task set "
                "(finance_world.v1 or finance_world.v2)"
            )
        if set(self.by_difficulty) != set(Difficulty):
            raise ValueError("by_difficulty must contain every Difficulty exactly once")
        if sum(self.by_task.values()) != sum(self.by_difficulty.values()):
            raise ValueError("task and difficulty count totals must match")
        return self

    def require_finance_world(
        self,
        finance_world: FinanceWorldVersion,
    ) -> TaskDifficultyCounts:
        """Require the task set selected by an explicit finance-world version."""
        expected = _TASK_IDS_BY_FINANCE_WORLD[finance_world]
        actual = set(self.by_task)
        if actual != expected:
            missing = sorted(task.value for task in expected - actual)
            extra = sorted(task.value for task in actual - expected)
            raise ValueError(
                f"by_task must contain exactly the {finance_world} task set; "
                f"missing={missing}, extra={extra}"
            )
        return self


class Dataset(FrozenModel):
    """Immutable public Dataset resource."""

    schema_version: Literal["distillery.dataset.v1"] = "distillery.dataset.v1"
    dataset_id: DatasetId
    content_sha256: Sha256Hex
    split_sha256: SplitHashes
    uri: StrictStr = Field(min_length=1)
    provenance_summary: StrictStr = Field(min_length=1)
    task_difficulty_counts: TaskDifficultyCounts
    example_count: NonNegativeSafeInt
    created_at: AwareDatetime
    metadata: FrozenJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _example_count_matches_breakdowns(self) -> Dataset:
        task_total = sum(self.task_difficulty_counts.by_task.values())
        difficulty_total = sum(self.task_difficulty_counts.by_difficulty.values())
        if self.example_count != task_total or self.example_count != difficulty_total:
            raise ValueError(
                "example_count must equal both task and difficulty count totals"
            )
        return self

    def resource_hash(self) -> str:
        """Hash-address of the resource excluding mutable wall-clock fields."""
        payload = self.model_dump(mode="python", exclude={"created_at"})
        return content_sha256(payload)
