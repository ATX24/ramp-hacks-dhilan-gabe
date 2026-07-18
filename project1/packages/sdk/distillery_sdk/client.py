"""Distillery Python SDK — the three-line happy path:

    distillery = Distillery(base_url=...)
    dataset = distillery.datasets.create("./finance_world.jsonl")
    run = distillery.distill(dataset, recipe="auto").wait()
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx


class SDKError(Exception):
    def __init__(self, payload: dict):
        self.payload = payload
        super().__init__(payload.get("error", payload))


class DistillationRun:
    def __init__(self, client: "Distillery", doc: dict):
        self._client = client
        self.doc = doc

    @property
    def run_id(self) -> str:
        return self.doc["run_id"]

    @property
    def state(self) -> str:
        return self.doc["state"]

    def refresh(self) -> "DistillationRun":
        self.doc = self._client._get(f"/v1/distillation-runs/{self.run_id}")
        return self

    def wait(self, poll_seconds: float = 5.0, timeout: float = 14400) -> "DistillationRun":
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.refresh()
            if self.state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                return self
            time.sleep(poll_seconds)
        raise TimeoutError(f"run {self.run_id} still {self.state}")

    def cancel(self) -> "DistillationRun":
        self.doc = self._client._post(f"/v1/distillation-runs/{self.run_id}/cancel", {})
        return self

    @property
    def model_artifact(self):
        return self.doc.get("model_artifact_id")

    @property
    def proof_report(self):
        return self.doc.get("proof_report_id")


class _Datasets:
    def __init__(self, client: "Distillery"):
        self._client = client

    def create(self, path: str | Path) -> dict:
        p = Path(path)
        with p.open("rb") as f:
            r = self._client._http.post("/v1/datasets", files={"file": (p.name, f, "application/jsonl")})
        return self._client._unwrap(r)

    def generate(self, corpus: str = "smoke") -> dict:
        return self._client._post("/v1/datasets/generate", {"corpus": corpus})

    def get(self, dataset_id: str) -> dict:
        return self._client._get(f"/v1/datasets/{dataset_id}")

    def synthesize(self, dataset_id: str, mode: str = "teacher", dry_run: bool = True,
                   max_cost_usd: float = 25.0) -> dict:
        return self._client._post(f"/v1/datasets/{dataset_id}/synthesize",
                                  {"dataset_id": dataset_id, "mode": mode,
                                   "dry_run": dry_run, "max_cost_usd": max_cost_usd})

    def run_recipe(self, dataset_id: str, recipe: str, dry_run: bool = False,
                   max_cost_usd: float = 25.0) -> dict:
        """Run a user-defined synthesis recipe; returns stats and (unless
        dry_run) the new immutable dataset id in `new_dataset_id`."""
        return self._client._post(f"/v1/datasets/{dataset_id}/recipes/{recipe}",
                                  {"dry_run": dry_run, "max_cost_usd": max_cost_usd})


class Distillery:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None,
                 timeout: float = 120.0):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        self.datasets = _Datasets(self)

    def _unwrap(self, r: httpx.Response) -> dict:
        if r.status_code >= 400:
            raise SDKError(r.json())
        return r.json()

    def _get(self, path: str) -> dict:
        return self._unwrap(self._http.get(path))

    def _post(self, path: str, body: dict) -> dict:
        return self._unwrap(self._http.post(path, json=body))

    def plan(self, dataset: dict | str, recipe: str = "auto", max_run_usd: float = 25.0) -> dict:
        ds = dataset if isinstance(dataset, str) else dataset["dataset_id"]
        return self._post("/v1/distillation-runs/plan",
                          {"dataset_id": ds, "recipe": recipe, "max_run_usd": max_run_usd})

    def distill(self, dataset: dict | str, recipe: str = "auto", **kwargs) -> DistillationRun:
        ds = dataset if isinstance(dataset, str) else dataset["dataset_id"]
        doc = self._post("/v1/distillation-runs", {"dataset_id": ds, "recipe": recipe, **kwargs})
        return DistillationRun(self, doc)

    def recipes(self) -> dict:
        return self._get("/v1/recipes")
