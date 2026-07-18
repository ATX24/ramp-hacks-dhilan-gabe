import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from distillery.data.generate import generate_corpus, generate_split
from distillery.data.validate import validate_output
from distillery.contracts.dataset import canonical_json
from distillery.recipes.auto import resolve
from distillery.contracts.errors import RecipeNotImplemented, RecipeIncompatible


def test_generation_deterministic_and_invariants(tmp_path):
    m1 = generate_corpus({"train": 40, "validation": 10, "test_iid": 10, "test_ood": 10}, tmp_path / "a")
    m2 = generate_corpus({"train": 40, "validation": 10, "test_iid": 10, "test_ood": 10}, tmp_path / "b")
    assert m1["splits"]["train"]["sha256"] == m2["splits"]["train"]["sha256"]
    # oracle labels validate under their own deterministic validators
    for line in (tmp_path / "a" / "train.jsonl").read_text().splitlines():
        ex = json.loads(line)
        obj, errs = validate_output(ex["task"], canonical_json(ex["expected_output"]), ex["input"])
        assert errs == [], (ex["task"], errs)


def test_split_isolation():
    train, off = generate_split("train", 40, 0)
    val, _ = generate_split("validation", 10, off)
    assert {e.world_id for e in train}.isdisjoint({e.world_id for e in val})


def test_auto_resolver():
    r = resolve("auto", has_valid_responses=True, teacher_access="api_black_box",
                tokenizers_match=False, memory_dry_run_ok=False, teacher_available=False)
    assert r.resolved == "sequence.v1"
    r = resolve("auto", has_valid_responses=False, teacher_access="api_black_box",
                tokenizers_match=False, memory_dry_run_ok=False, teacher_available=True)
    assert r.resolved == "sequence.v1"
    with pytest.raises(RecipeIncompatible):
        resolve("logit.v1", has_valid_responses=True, teacher_access="api_black_box",
                tokenizers_match=False, memory_dry_run_ok=False, teacher_available=True)
    with pytest.raises(RecipeNotImplemented):
        resolve("gkd.on_policy.v0", has_valid_responses=True, teacher_access="api_black_box",
                tokenizers_match=False, memory_dry_run_ok=False, teacher_available=True)
    r = resolve("auto", has_valid_responses=False, teacher_access="api_black_box",
                tokenizers_match=False, memory_dry_run_ok=False, teacher_available=True,
                baseline_meets_gate=True)
    assert r.do_not_distill and r.resolved is None


def test_api_happy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTILLERY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    import importlib
    import apps.api.distillery_api.store as store_mod
    store_mod.DATA_DIR = Path(str(tmp_path))
    import apps.api.distillery_api.main as main_mod
    importlib.reload(main_mod)
    client = TestClient(main_mod.app)

    assert client.get("/healthz").json()["ok"]
    ds = client.post("/v1/datasets/generate", json={"corpus": "smoke"}).json()
    assert ds["counts_by_split"]["train"] == 320
    mix = ds["counts_by_task"]
    total = sum(ds["counts_by_split"].values())
    assert abs(mix["transaction_review"] / total - 0.45) < 0.02

    plan = client.post("/v1/distillation-runs/plan",
                       json={"dataset_id": ds["dataset_id"], "recipe": "auto"}).json()
    assert plan["recipe"]["resolved"] in ("sequence.v1", None)

    # oracle synthesis -> new labeled dataset -> run creation
    syn = client.post(f"/v1/datasets/{ds['dataset_id']}/synthesize",
                      json={"dataset_id": ds["dataset_id"], "mode": "oracle", "dry_run": False}).json()
    labeled = syn["new_dataset_id"]
    run = client.post("/v1/distillation-runs",
                      json={"dataset_id": labeled, "recipe": "sequence.v1"}).json()
    assert run["state"] == "QUEUED" and run["manifest_sha256"].startswith("sha256:")
    cancelled = client.post(f"/v1/distillation-runs/{run['run_id']}/cancel").json()
    assert cancelled["state"] == "CANCELLED"
    # idempotent cancel
    assert client.post(f"/v1/distillation-runs/{run['run_id']}/cancel").json()["state"] == "CANCELLED"

    # requesting an unimplemented recipe fails loudly
    bad = client.post("/v1/distillation-runs/plan",
                      json={"dataset_id": ds["dataset_id"], "recipe": "gkd.on_policy.v0"})
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "RECIPE_NOT_IMPLEMENTED"
