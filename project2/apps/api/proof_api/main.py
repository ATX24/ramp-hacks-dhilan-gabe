"""Proof control-plane API. Creation returns immediately; events stream via SSE."""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import proof  # noqa: F401  (registers built-in recipes)
from proof.resources import Target
from proof.service import ProofService


def build_backend():
    if os.environ.get("PROOF_BACKEND", "mock") == "bedrock":
        from proof.backends.bedrock import BedrockBackend
        return BedrockBackend(profile=os.environ.get("AWS_PROFILE"),
                              role_arn=os.environ["PROOF_BEDROCK_ROLE_ARN"],
                              bucket=os.environ["PROOF_BUCKET"])
    from proof.backends.mock import MockBackend
    return MockBackend()


app = FastAPI(title="Proof")
svc = ProofService(build_backend())


class DatasetIn(BaseModel):
    records: list[dict]


class DistillIn(BaseModel):
    teacher: str = "amazon:nova-pro"
    student: str = "amazon:nova-micro"
    dataset_id: str
    recipe: str = "managed-distillation"
    target: Target = Target()


class DeployIn(BaseModel):
    model_id: str
    alias: str
    fallback: str = "amazon:nova-pro"


class InvokeIn(BaseModel):
    input: str
    force_fallback: bool = False


@app.post("/v1/datasets")
def create_dataset(body: DatasetIn):
    ds = svc.create_dataset(body.records)
    return {"id": ds.id, "n": len(ds.records), "fingerprint": ds.fingerprint}


@app.post("/v1/distillations")
def create_distillation(body: DistillIn):
    try:
        d = svc.create_distillation(body.teacher, body.student, body.dataset_id,
                                    body.recipe, body.target)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return d.model_dump()


@app.get("/v1/distillations/{dist_id}")
def get_distillation(dist_id: str):
    d = svc.distillations.get(dist_id)
    if not d:
        raise HTTPException(404, "not found")
    return d.model_dump()


@app.get("/v1/distillations/{dist_id}/events")
async def distillation_events(dist_id: str):
    if dist_id not in svc.events:
        raise HTTPException(404, "not found")

    async def gen():
        q = svc.events[dist_id]
        while True:
            evt = await q.get()
            yield {"data": json.dumps(evt)}
            if evt.get("event") == "done":
                return

    return EventSourceResponse(gen())


@app.get("/v1/recipes")
def list_recipes():
    from proof.recipes.base import registered
    return {"recipes": registered()}


@app.post("/v1/deployments")
def deploy(body: DeployIn):
    try:
        dep = svc.promote(body.model_id, body.alias, body.fallback)
    except (KeyError, ValueError) as e:
        raise HTTPException(409, str(e))
    return dep.model_dump()


@app.post("/v1/deployments/{alias}/rollback")
def rollback(alias: str):
    try:
        return svc.rollback(alias).model_dump()
    except (KeyError, ValueError) as e:
        raise HTTPException(409, str(e))


@app.post("/v1/deployments/{alias}/invoke")
async def invoke(alias: str, body: InvokeIn):
    if alias not in svc.deployments:
        raise HTTPException(404, "unknown alias")
    return await svc.invoke(alias, body.input, body.force_fallback)
