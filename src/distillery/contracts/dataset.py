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
        if set(self.by_task) != set(TaskId):
            raise ValueError("by_task must contain every executable TaskId exactly once")
        if set(self.by_difficulty) != set(Difficulty):
            raise ValueError("by_difficulty must contain every Difficulty exactly once")
        if sum(self.by_task.values()) != sum(self.by_difficulty.values()):
            raise ValueError("task and difficulty count totals must match")
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
