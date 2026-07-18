"""Proof SDK — the three-line happy path from the shared plan:

    proof = Proof(base_url=...)
    job = await proof.distill(teacher="amazon:nova-pro", data="./traces.jsonl",
                              target={"quality": 0.92, "cost": 0.1})
    model = await job.deploy("merchant-normalizer")
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

TERMINAL = {"READY", "BLOCKED", "FAILED"}


class DistillationJob:
    def __init__(self, client: httpx.AsyncClient, data: dict):
        self._c = client
        self.id = data["id"]
        self.status = data["status"]
        self.report: dict | None = None
        self.model_id: str | None = None

    async def wait(self, poll: float = 0.2) -> "DistillationJob":
        while self.status not in TERMINAL:
            await asyncio.sleep(poll)
            r = (await self._c.get(f"/v1/distillations/{self.id}")).json()
            self.status, self.report, self.model_id = r["status"], r.get("report"), r.get("model_id")
        return self

    async def deploy(self, alias: str, fallback: str = "amazon:nova-pro") -> dict:
        await self.wait()
        if self.status != "READY":
            raise RuntimeError(f"distillation {self.id} is {self.status}, not READY "
                               f"(report={self.report})")
        r = await self._c.post("/v1/deployments", json={
            "model_id": self.model_id, "alias": alias, "fallback": fallback})
        r.raise_for_status()
        return r.json()


class Proof:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None):
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self._c = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=60)

    async def distill(self, *, teacher: str, data: str | list[dict],
                      target: dict[str, Any] | None = None,
                      student: str = "amazon:nova-micro",
                      recipe: str = "managed-distillation") -> DistillationJob:
        records = ([json.loads(l) for l in Path(data).read_text().splitlines() if l.strip()]
                   if isinstance(data, str) else data)
        ds = (await self._c.post("/v1/datasets", json={"records": records})).json()
        r = await self._c.post("/v1/distillations", json={
            "teacher": teacher, "student": student, "dataset_id": ds["id"],
            "recipe": recipe, "target": target or {}})
        r.raise_for_status()
        return DistillationJob(self._c, r.json())

    async def invoke(self, alias: str, input: str, force_fallback: bool = False) -> dict:
        r = await self._c.post(f"/v1/deployments/{alias}/invoke",
                               json={"input": input, "force_fallback": force_fallback})
        r.raise_for_status()
        return r.json()

    async def rollback(self, alias: str) -> dict:
        r = await self._c.post(f"/v1/deployments/{alias}/rollback")
        r.raise_for_status()
        return r.json()
