"""User-defined recipe surface: registry, resolver gating, catalog, endpoint."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_tmp = tempfile.mkdtemp(prefix="distillery-test-")
os.environ["DISTILLERY_DATA_DIR"] = _tmp
import apps.api.distillery_api.store as store_mod

store_mod.DATA_DIR = Path(_tmp)

from apps.api.distillery_api.main import app
from distillery.contracts.errors import RecipeIncompatible
from distillery.recipes.auto import resolve
from distillery.recipes.custom import (SynthesisContext, custom_names, get_custom,
                                       register)

client = TestClient(app)


def _dataset_id() -> str:
    r = client.post("/v1/datasets/generate", json={"corpus": "smoke"})
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


def test_builtin_customs_registered_and_cataloged():
    assert {"rejection_sampling.v1", "oracle_curriculum.v1"} <= set(custom_names())
    names = {r["name"]: r for r in client.get("/v1/recipes").json()["recipes"]}
    assert names["oracle_curriculum.v1"]["implemented"] is True
    assert "user-defined" in names["oracle_curriculum.v1"]["signal"]


def test_resolver_accepts_custom_and_gates_requirements():
    res = resolve("oracle_curriculum.v1", has_valid_responses=False,
                  teacher_access="api_black_box", tokenizers_match=False,
                  memory_dry_run_ok=False, teacher_available=False)
    assert res.resolved == "oracle_curriculum.v1"

    with pytest.raises(RecipeIncompatible):
        resolve("rejection_sampling.v1", has_valid_responses=False,
                teacher_access="api_black_box", tokenizers_match=False,
                memory_dry_run_ok=False, teacher_available=False)


def test_recipe_endpoint_produces_new_immutable_dataset():
    ds = _dataset_id()
    r = client.post(f"/v1/datasets/{ds}/recipes/oracle_curriculum.v1",
                    json={"dry_run": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["output_examples"] > body["input_examples"]  # hard oversampling
    assert body["new_dataset_id"] != ds
    new_meta = client.get(f"/v1/datasets/{body['new_dataset_id']}").json()
    assert new_meta["counts_by_label_source"].get("oracle", 0) > 0
    assert new_meta["source"].startswith("recipe:oracle_curriculum.v1:")


def test_unknown_recipe_404s_and_teacher_recipe_blocked_without_key(monkeypatch):
    ds = _dataset_id()
    r = client.post(f"/v1/datasets/{ds}/recipes/nope.v9", json={})
    assert r.status_code == 422 and "RECIPE_NOT_IMPLEMENTED" in r.text
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post(f"/v1/datasets/{ds}/recipes/rejection_sampling.v1", json={})
    assert r.status_code in (409, 422, 400)
    assert "RECIPE_INCOMPATIBLE" in r.text or "requires the teacher" in r.text


def test_user_registered_recipe_runs_through_endpoint():
    class KeepEasy:
        name = "keep_easy.test"
        requires = frozenset()
        description = "test-only: oracle-label then keep easy examples"

        def run(self, ctx: SynthesisContext, examples):
            ctx.oracle_label(examples)
            kept = [ex for ex in examples if ex.difficulty == "easy"]
            ctx.emit("kept_easy", n=len(kept))
            return kept

    if get_custom("keep_easy.test") is None:
        register(KeepEasy())
    ds = _dataset_id()
    r = client.post(f"/v1/datasets/{ds}/recipes/keep_easy.test", json={"dry_run": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["output_examples"] < body["input_examples"]
    assert body["events"] and body["events"][0]["event"] == "kept_easy"
