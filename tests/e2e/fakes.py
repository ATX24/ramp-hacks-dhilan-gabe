"""In-memory Distillery fakes for Curate → Synthesize → Train-planned → Prove.

No network, no GPU, no SageMaker. Exercises the high-level stage contract that
examples and the future SDK share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


class FakeTrainingLaunchError(RuntimeError):
    pass


@dataclass
class FakeDataset:
    dataset_id: str
    uri: str
    content_sha256: str
    example_count: int
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FakePlan:
    requested_recipe: str
    resolved_recipe: str | None
    resolver_reasons: tuple[str, ...]
    do_not_distill: bool
    teacher_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    student_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    teacher_revision: str = "pending_preflight_sha"
    student_revision: str = "pending_preflight_sha"
    estimate_low_usd: float = 0.0
    estimate_high_usd: float = 25.0
    blockers: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    launched_training: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_recipe": self.requested_recipe,
            "resolved_recipe": self.resolved_recipe,
            "resolver_reasons": list(self.resolver_reasons),
            "do_not_distill": self.do_not_distill,
            "teacher": {"id": self.teacher_id, "revision": self.teacher_revision},
            "student": {"id": self.student_id, "revision": self.student_revision},
            "cost_estimate_usd": {
                "low": self.estimate_low_usd,
                "high": self.estimate_high_usd,
            },
            "blockers": list(self.blockers),
            "details": self.details,
            "launches_training": self.launched_training,
        }


@dataclass
class FakeProofReport:
    report_id: str
    proof_status: str
    limitations: tuple[str, ...]
    run_ids: tuple[str, ...] = ()


@dataclass
class FakeRun:
    run_id: str
    dataset_id: str
    state: str
    resolved_recipe: str | None
    model_artifact_id: str | None = None
    proof_report_id: str | None = None

    def wait(self) -> FakeRun:
        return self


class FakeDatasetsAPI:
    def __init__(self, parent: FakeDistillery) -> None:
        self._parent = parent

    def create(self, path: str | Path) -> FakeDataset:
        return self._parent.create_dataset(Path(path))


class FakeDistillery:
    """Minimal client implementing the demo-facing Distillery surface."""

    def __init__(self, *, allow_train: bool = False) -> None:
        self.allow_train = allow_train
        self.datasets = FakeDatasetsAPI(self)
        self.plan_calls = 0
        self.distill_calls = 0
        self._last_rows: list[dict[str, Any]] = []

    def create_dataset(self, path: Path) -> FakeDataset:
        import json

        text = path.read_text(encoding="utf-8")
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
        digest = sha256(path.read_bytes()).hexdigest()
        self._last_rows = rows
        return FakeDataset(
            dataset_id=f"ds_fake_{digest[:10]}",
            uri=path.resolve().as_uri(),
            content_sha256=digest,
            example_count=len(rows),
            rows=rows,
        )

    def plan_distillation(
        self, dataset: FakeDataset, *, recipe: str = "auto"
    ) -> FakePlan:
        self.plan_calls += 1
        usable = 0
        for row in dataset.rows or self._last_rows:
            source = row.get("provenance", {}).get("label_source")
            if source in {"oracle", "imported", "teacher"}:
                usable += 1

        resolved: str | None = "sequence.v1"
        reasons: tuple[str, ...] = ("usable_responses_present", "fake_plan")
        do_not_distill = False

        try:
            from distillery.contracts.recipes import (
                AutoResolverInput,
                resolve_requested_recipe,
            )

            result = resolve_requested_recipe(
                recipe,
                auto_input=AutoResolverInput(usable_responses_exist=usable > 0),
            )
            resolved = result.resolved
            reasons = result.reasons
            do_not_distill = resolved == "do_not_distill"
        except ImportError:
            if recipe not in {"auto", "sequence.v1", "logit.v1"}:
                resolved = None
                reasons = ("recipe_unavailable_without_contracts",)

        return FakePlan(
            requested_recipe=recipe,
            resolved_recipe=resolved,
            resolver_reasons=reasons,
            do_not_distill=do_not_distill,
            details={
                "dataset_id": dataset.dataset_id,
                "usable_responses": usable,
                "planned_at": datetime.now(UTC).isoformat(),
            },
        )

    def distill(
        self,
        dataset: FakeDataset,
        *,
        recipe: str = "auto",
        training_acknowledged: bool = False,
    ) -> FakeRun:
        self.distill_calls += 1
        if not training_acknowledged:
            raise FakeTrainingLaunchError("missing training acknowledgment")
        if not self.allow_train:
            raise FakeTrainingLaunchError(
                "FakeDistillery.allow_train is False; e2e suite must not train"
            )
        plan = self.plan_distillation(dataset, recipe=recipe)
        return FakeRun(
            run_id="run_fake_001",
            dataset_id=dataset.dataset_id,
            state="QUEUED",
            resolved_recipe=plan.resolved_recipe,
            model_artifact_id=None,
            proof_report_id=None,
        )

    def prove_from_plan(self, plan: FakePlan, *, run_id: str = "run_planned") -> FakeProofReport:
        status = "do_not_distill" if plan.do_not_distill else "insufficient_evidence"
        return FakeProofReport(
            report_id="prf_fake_001",
            proof_status=status,
            run_ids=(run_id,),
            limitations=(
                "E2E fake proof; real benchmarks pending.",
                "No training was executed.",
            ),
        )
