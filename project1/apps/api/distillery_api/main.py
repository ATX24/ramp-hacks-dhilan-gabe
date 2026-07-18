"""Distillery control plane. Thin FastAPI app: validates and seals manifests,
stores immutable resources, runs teacher synthesis. It does not host models or
execute GPU training."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from distillery import __version__
from distillery.contracts.dataset import Example, DatasetMeta, hash_examples, canonical_json
from distillery.contracts.errors import DistilleryError, InvalidDataset, NotFound, Cancelled
from distillery.contracts.manifest import (RunManifest, DatasetRef, ModelPin, TrainingConfig,
                                           RuntimeConfig, CostConfig, ALLOWED_TRANSITIONS)
from distillery.contracts.recipes import catalog
from distillery.contracts.errors import RecipeNotImplemented
from distillery.recipes.custom import (SynthesisContext, check_requirements,
                                       get_custom, load_plugins)
from distillery.data.generate import generate_corpus, SMOKE_PLAN, FULL_PLAN
from distillery.synthesis.teacher import synthesize_missing, materialize_oracle_responses

from .planner import plan as plan_distillation, STUDENT, TEACHER
from .store import Store, new_id, now_iso

app = FastAPI(title="Distillery", version=__version__,
              description="Curate -> Synthesize -> Train -> Prove. Smaller models. Proven economics.")
store = Store()
LOADED_RECIPE_PLUGINS = load_plugins()  # DISTILLERY_RECIPES=my_pkg.my_recipes,...


@app.exception_handler(DistilleryError)
async def handle_distillery_error(_: Request, exc: DistilleryError):
    return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})


@app.get("/")
def root():
    return {"service": "distillery", "version": __version__,
            "tagline": "Smaller models. Proven economics.",
            "workflow": ["curate", "synthesize", "train", "prove"]}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1/recipes")
def list_recipes():
    return {"recipes": [r.model_dump() for r in catalog()]}


# ---------- Datasets (Curate) ----------

def _ingest_examples(examples: list[Example], source: str) -> dict:
    if not examples:
        raise InvalidDataset("Dataset contains no examples.")
    dataset_id = new_id("ds")
    blob = store.blob_dir("datasets", dataset_id)
    split_lines: dict[str, list[str]] = {}
    for ex in examples:
        split_lines.setdefault(ex.provenance.split, []).append(canonical_json(ex.model_dump()))
    split_sha = {}
    for split, lines in split_lines.items():
        (blob / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
        split_sha[split] = hash_examples([e for e in examples if e.provenance.split == split])
    meta = DatasetMeta(
        dataset_id=dataset_id, sha256=hash_examples(examples), split_sha256=split_sha,
        counts_by_task=_count(examples, lambda e: e.task),
        counts_by_split=_count(examples, lambda e: e.provenance.split),
        counts_by_label_source=_count(
            examples, lambda e: e.provenance.label_source if e.response else "unlabeled"),
        counts_by_difficulty=_count(examples, lambda e: e.difficulty),
        uri=f"store://datasets/{dataset_id}", created_at=now_iso(),
    )
    doc = meta.model_dump() | {"source": source}
    store.put("datasets", dataset_id, doc)
    return doc


def _count(exs, key):
    d: dict[str, int] = {}
    for e in exs:
        d[key(e)] = d.get(key(e), 0) + 1
    return d


@app.post("/v1/datasets")
async def create_dataset(file: UploadFile):
    raw = (await file.read()).decode()
    examples = []
    for i, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        try:
            examples.append(Example.model_validate_json(line))
        except Exception as e:
            raise InvalidDataset(f"Line {i + 1} failed schema validation.", details={"error": str(e)})
    return _ingest_examples(examples, source="upload")


class GenerateReq(BaseModel):
    corpus: str = "smoke"  # smoke | full


@app.post("/v1/datasets/generate")
def generate_dataset(req: GenerateReq):
    plan = SMOKE_PLAN if req.corpus == "smoke" else FULL_PLAN
    tmp = store.root / "tmp_gen"
    generate_corpus(plan, tmp)
    examples = []
    for f in sorted(tmp.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            examples.append(Example.model_validate_json(line))
    return _ingest_examples(examples, source=f"synthetic:{req.corpus}")


@app.get("/v1/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    return store.get("datasets", dataset_id)


# ---------- Synthesize ----------

class SynthesizeReq(BaseModel):
    dataset_id: str
    mode: str = "teacher"  # teacher | oracle
    dry_run: bool = True
    max_cost_usd: float = 25.0


@app.post("/v1/datasets/{dataset_id}/synthesize")
def synthesize(dataset_id: str, req: SynthesizeReq):
    meta = store.get("datasets", dataset_id)
    blob = store.blob_dir("datasets", dataset_id)
    examples: list[Example] = []
    for split in ("train", "validation"):
        p = blob / f"{split}.jsonl"
        if p.exists():
            examples += [Example.model_validate_json(l) for l in p.read_text().splitlines() if l.strip()]
    if req.mode == "oracle":
        n = materialize_oracle_responses(examples)
        stats = {"mode": "oracle", "materialized": n}
    else:
        stats = synthesize_missing(examples, blob / "synthesis_responses.jsonl",
                                   max_cost_usd=req.max_cost_usd, dry_run=req.dry_run)
        stats["mode"] = "teacher"
    if not req.dry_run or req.mode == "oracle":
        # A labeled dataset is a NEW immutable dataset.
        new_doc = _ingest_examples(
            examples + _load_test_splits(blob), source=f"synthesized:{req.mode}:{dataset_id}")
        stats["new_dataset_id"] = new_doc["dataset_id"]
    return stats


class RecipeRunReq(BaseModel):
    dry_run: bool = False
    max_cost_usd: float = 25.0


@app.post("/v1/datasets/{dataset_id}/recipes/{recipe_name}")
def run_custom_recipe(dataset_id: str, recipe_name: str, req: RecipeRunReq):
    """Run a user-defined synthesis recipe over train/validation examples and
    ingest its curated output as a NEW immutable dataset. Test splits are
    carried over untouched — the recipe never sees them."""
    recipe = get_custom(recipe_name)
    if recipe is None:
        raise RecipeNotImplemented(f"Unknown custom recipe '{recipe_name}'.",
                                   details={"recipe": recipe_name})
    check_requirements(recipe, teacher_available=bool(os.environ.get("ANTHROPIC_API_KEY")))

    store.get("datasets", dataset_id)  # 404 if missing
    blob = store.blob_dir("datasets", dataset_id)
    examples: list[Example] = []
    for split in ("train", "validation"):
        p = blob / f"{split}.jsonl"
        if p.exists():
            examples += [Example.model_validate_json(l) for l in p.read_text().splitlines() if l.strip()]

    events: list[dict] = []
    ctx = SynthesisContext(out_dir=blob, max_cost_usd=req.max_cost_usd,
                           dry_run=req.dry_run, emit=events.append)
    kept = recipe.run(ctx, examples)

    result = {"recipe": recipe_name, "input_examples": len(examples),
              "output_examples": len(kept), "events": events,
              "teacher_stats": ctx.teacher_stats, "dry_run": req.dry_run}
    if not req.dry_run:
        new_doc = _ingest_examples(kept + _load_test_splits(blob),
                                   source=f"recipe:{recipe_name}:{dataset_id}")
        result["new_dataset_id"] = new_doc["dataset_id"]
    return result


def _load_test_splits(blob: Path) -> list[Example]:
    out = []
    for split in ("test_iid", "test_ood"):
        p = blob / f"{split}.jsonl"
        if p.exists():
            out += [Example.model_validate_json(l) for l in p.read_text().splitlines() if l.strip()]
    return out


# ---------- Plan + Runs (Train) ----------

class PlanReq(BaseModel):
    dataset_id: str
    recipe: str = "auto"
    max_run_usd: float = 25.0


@app.post("/v1/distillation-runs/plan")
def plan_run(req: PlanReq):
    meta = store.get("datasets", req.dataset_id)
    return plan_distillation(meta, req.recipe, req.max_run_usd)


class CreateRunReq(BaseModel):
    dataset_id: str
    recipe: str = "auto"
    arm: str = "sequence_kd"
    seed: int = 17
    max_steps: int = 200
    max_run_usd: float = 25.0
    backend: str = "local"


@app.post("/v1/distillation-runs")
def create_run(req: CreateRunReq):
    meta = store.get("datasets", req.dataset_id)
    p = plan_distillation(meta, req.recipe, req.max_run_usd)
    if p["blockers"]:
        raise InvalidDataset("Preflight blockers must be resolved before submission.",
                             details={"blockers": p["blockers"]})
    if p["recipe"].get("do_not_distill"):
        raise InvalidDataset("Resolver recommends do_not_distill; no run submitted.",
                             details={"resolution": p["recipe"]})
    run_id = new_id("run")
    manifest = RunManifest(
        run_id=run_id, created_at=now_iso(),
        dataset=DatasetRef(dataset_id=req.dataset_id, uri=meta["uri"], sha256=meta["sha256"],
                           split_sha256=meta["split_sha256"]),
        models={"teacher": ModelPin(**TEACHER), "student": ModelPin(**STUDENT)},
        recipe=p["recipe"], arm=req.arm,
        training=TrainingConfig(seed=req.seed, max_steps=req.max_steps),
        runtime=RuntimeConfig(backend=req.backend),
        cost=CostConfig(max_run_usd=req.max_run_usd,
                        estimate_low_usd=p["estimates"]["teacher_cost_low_usd"],
                        estimate_high_usd=p["estimates"]["teacher_cost_high_usd"]),
        output_prefix=f"store://runs/{run_id}/",
    )
    doc = {"run_id": run_id, "state": "QUEUED", "manifest": manifest.model_dump(),
           "manifest_sha256": manifest.seal_hash(), "created_at": now_iso(),
           "model_artifact_id": None, "proof_report_id": None, "failure": None}
    store.put("runs", run_id, doc)
    store.append_event(run_id, {"state": "QUEUED"})
    return doc


@app.get("/v1/distillation-runs/{run_id}")
def get_run(run_id: str):
    return store.get("runs", run_id)


@app.post("/v1/distillation-runs/{run_id}/cancel")
def cancel_run(run_id: str):
    doc = store.get("runs", run_id)
    if doc["state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
        return doc  # idempotent
    doc["state"] = "CANCELLED"
    doc["failure"] = Cancelled("Cancelled by user.", run_id=run_id).to_dict()
    store.put("runs", run_id, doc, overwrite=True)
    store.append_event(run_id, {"state": "CANCELLED"})
    return doc


@app.get("/v1/model-artifacts/{artifact_id}")
def get_artifact(artifact_id: str):
    return store.get("artifacts", artifact_id)


@app.get("/v1/proof-reports/{report_id}")
def get_report(report_id: str):
    return store.get("reports", report_id)
