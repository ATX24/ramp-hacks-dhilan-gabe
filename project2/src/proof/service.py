"""Proof control plane: resource store + distillation orchestration.

Creation returns immediately; the run happens in a background task; progress
streams through per-distillation event queues (SSE at the API layer).
"""
from __future__ import annotations

import asyncio
import traceback

from proof.backends.base import Backend, RecipeIncompatible
from proof.gate import prove
from proof.metrics import schema_valid
from proof.recipes.base import Recipe, RecipeContext, get_recipe
from proof.resources import (Dataset, Deployment, Distillation,
                             DistillationStatus, Model, Record, Target)


class ProofService:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.datasets: dict[str, Dataset] = {}
        self.distillations: dict[str, Distillation] = {}
        self.models: dict[str, Model] = {}
        self.deployments: dict[str, Deployment] = {}  # keyed by alias
        self.events: dict[str, asyncio.Queue] = {}

    # -- resources -----------------------------------------------------------
    def create_dataset(self, records: list[dict]) -> Dataset:
        ds = Dataset(records=[Record(**r) if isinstance(r, dict) else r for r in records])
        self.datasets[ds.id] = ds
        return ds

    def create_distillation(self, teacher: str, student_base: str, dataset_id: str,
                            recipe: str, target: Target,
                            recipe_obj: Recipe | None = None) -> Distillation:
        if dataset_id not in self.datasets:
            raise KeyError(f"unknown dataset {dataset_id}")
        recipe_impl = recipe_obj or get_recipe(recipe)  # validate before queueing
        dist = Distillation(teacher=teacher, student_base=student_base,
                            dataset_id=dataset_id, recipe=recipe_impl.name, target=target)
        self.distillations[dist.id] = dist
        self.events[dist.id] = asyncio.Queue()
        asyncio.get_event_loop().create_task(self._run(dist, recipe_impl))
        return dist

    def promote(self, model_id: str, alias: str, fallback: str) -> Deployment:
        model = self.models[model_id]
        if not model.report.get("passed"):
            raise ValueError(f"model {model_id} did not pass the proof gate; refusing to deploy")
        existing = self.deployments.get(alias)
        if existing:
            existing.history.append(existing.model_id)
            existing.model_id = model_id
            return existing
        dep = Deployment(alias=alias, model_id=model_id, fallback=fallback)
        self.deployments[alias] = dep
        return dep

    def rollback(self, alias: str) -> Deployment:
        dep = self.deployments[alias]
        if not dep.history:
            raise ValueError("no prior model to roll back to")
        dep.model_id = dep.history.pop()
        return dep

    # -- invocation with fallback -------------------------------------------
    async def invoke(self, alias: str, descriptor: str, force_fallback: bool = False) -> dict:
        dep = self.deployments[alias]
        model = self.models[dep.model_id]
        served_by = model.backend_ref
        out = None if force_fallback else (await self.backend.sample(served_by, [descriptor]))[0]
        fell_back = False
        if not schema_valid(out):
            out = (await self.backend.sample(dep.fallback, [descriptor]))[0]
            served_by, fell_back = dep.fallback, True
        return {"alias": alias, "output": out, "served_by": served_by, "fallback": fell_back}

    # -- orchestration -------------------------------------------------------
    def _emit(self, dist_id: str, payload: dict) -> None:
        self.events[dist_id].put_nowait(payload)

    async def _run(self, dist: Distillation, recipe: Recipe) -> None:
        emit = lambda p: self._emit(dist.id, p)
        try:
            dist.status = DistillationStatus.RUNNING
            emit({"event": "status", "status": "RUNNING", "recipe": recipe.name})

            ds = self.datasets[dist.dataset_id]
            train, holdout = ds.split()
            ctx = RecipeContext(self.backend, dist.teacher, dist.student_base,
                                lambda e: emit({**e, "phase": "recipe"}))
            candidate = await recipe.run(ctx, train)

            dist.status = DistillationStatus.EVALUATING
            emit({"event": "status", "status": "EVALUATING", "holdout": len(holdout)})
            report = await prove(self.backend, candidate, dist.teacher, holdout,
                                 dist.target, ctx.spend_usd)
            dist.report = report

            if report["passed"]:
                model = Model(base=candidate.base, backend_ref=candidate.ref,
                              distillation_id=dist.id, report=report)
                self.models[model.id] = model
                dist.model_id = model.id
                dist.status = DistillationStatus.READY
            else:
                dist.status = DistillationStatus.BLOCKED  # gate failed: no model ships
            emit({"event": "status", "status": dist.status.value, "report": report})
        except RecipeIncompatible as e:
            dist.status = DistillationStatus.FAILED
            dist.error = f"RECIPE_INCOMPATIBLE: {e}"
            emit({"event": "status", "status": "FAILED", "error": dist.error})
        except Exception as e:  # noqa: BLE001
            dist.status = DistillationStatus.FAILED
            dist.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            emit({"event": "status", "status": "FAILED", "error": dist.error})
        finally:
            emit({"event": "done"})
