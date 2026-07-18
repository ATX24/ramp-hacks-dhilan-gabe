"""Immutable Dataset resource contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    StrictStr,
    model_serializer,
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

TaskSetVersion = Literal["finance_world.v1", "finance_world.v2"]
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


class SplitHashes(FrozenModel):
    train: Sha256Hex
    validation: Sha256Hex
    test: Sha256Hex | None = None
    iid_test: Sha256Hex | None = None
    ood_test: Sha256Hex | None = None


class TaskDifficultyCounts(FrozenModel):
    task_set_version: TaskSetVersion = "finance_world.v1"
    by_task: FrozenDict[TaskId, NonNegativeSafeInt]
    by_difficulty: FrozenDict[Difficulty, NonNegativeSafeInt]

    @model_validator(mode="before")
    @classmethod
    def _infer_v2_for_legacy_unversioned_input(cls, value: Any) -> Any:
        """Infer v2 only when an unversioned wire already contains Merchant Tagging."""
        if not isinstance(value, Mapping) or "task_set_version" in value:
            return value
        by_task = value.get("by_task")
        if not isinstance(by_task, Mapping):
            return value
        task_values = {
            key.value if isinstance(key, TaskId) else str(key) for key in by_task
        }
        if TaskId.MERCHANT_TAGGING.value not in task_values:
            return value
        return {**value, "task_set_version": "finance_world.v2"}

    @model_validator(mode="after")
    def _complete_consistent_totals(self) -> TaskDifficultyCounts:
        required_tasks = (
            FINANCE_WORLD_V1_TASK_IDS
            if self.task_set_version == "finance_world.v1"
            else FINANCE_WORLD_V2_TASK_IDS
        )
        actual_tasks = set(self.by_task)
        if actual_tasks != required_tasks:
            missing = sorted(task.value for task in required_tasks - actual_tasks)
            extra = sorted(task.value for task in actual_tasks - required_tasks)
            raise ValueError(
                "by_task must contain every executable TaskId for "
                f"{self.task_set_version} exactly once; missing={missing}, extra={extra}"
            )
        if set(self.by_difficulty) != set(Difficulty):
            raise ValueError("by_difficulty must contain every Difficulty exactly once")
        if sum(self.by_task.values()) != sum(self.by_difficulty.values()):
            raise ValueError("task and difficulty count totals must match")
        return self

    @model_serializer(mode="wrap")
    def _serialize_versioned(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        payload = handler(self)
        if self.task_set_version == "finance_world.v1":
            payload.pop("task_set_version", None)
        return payload


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
