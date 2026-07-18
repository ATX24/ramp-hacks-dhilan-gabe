"""Routing, schema validation, and task coverage with FakeRuntime."""

from __future__ import annotations

import json

import pytest
from distillery_inference.errors import InferenceError, InferenceErrorCode
from distillery_inference.schemas import InferRequest
from distillery_inference.server import create_app
from distillery_inference.validation import validate_structured_output
from fastapi.testclient import TestClient
from support import make_service, sample_input


@pytest.mark.parametrize(
    "task",
    ["transaction_review", "variance_analysis", "cash_reconciliation"],
)
def test_infer_routes_each_finance_task(tmp_path, task: str) -> None:
    service, _bundle, _runtime = make_service(tmp_path)
    response = service.infer(
        InferRequest(
            model_id="model_sequence_kd",
            artifact_id="artifact_sequence_kd",
            task=task,  # type: ignore[arg-type]
            example_id=f"ex_{task}",
            input=sample_input(task),
        )
    )
    assert response.status == "ok"
    assert response.provenance == "live"
    assert response.task == task
    assert response.structured_output["task"] == task
    assert response.latency_ms >= 0
    assert response.prompt_tokens is not None
    assert response.completion_tokens is not None
    state, detail = validate_structured_output(task, response.structured_output)  # type: ignore[arg-type]
    assert state == "valid"
    assert detail is None


def test_http_invocations_and_demo_paths(tmp_path) -> None:
    service, _bundle, _runtime = make_service(tmp_path)
    client = TestClient(create_app(service=service))

    ping = client.get("/ping")
    assert ping.status_code == 200
    assert ping.text == "ok"

    health = client.get("/v1/demo/health")
    assert health.status_code == 200
    body = health.json()
    assert body["serving_ready"] is True
    assert "model_sequence_kd" in body["available_model_ids"]

    models = client.get("/v1/demo/models")
    assert models.status_code == 200
    registry = models.json()
    assert registry["schema_version"] == "distillery.demo_model_registry.v1"
    model_ids = {entry["model_id"] for entry in registry["models"]}
    assert {
        "model_student_base",
        "model_oracle_sft",
        "model_sequence_kd",
        "model_logit_kd",
        "model_ce_ablation",
        "model_promoted_winner",
    } <= model_ids

    payload = {
        "model_id": "model_oracle_sft",
        "artifact_id": "artifact_oracle_sft",
        "task": "transaction_review",
        "example_id": "ex_demo_txn_saas_001",
        "input": sample_input(),
    }
    invoked = client.post("/invocations", content=json.dumps(payload))
    assert invoked.status_code == 200
    assert invoked.json()["structured_output"]["task"] == "transaction_review"

    demo = client.post("/v1/demo/infer", json=payload)
    assert demo.status_code == 200
    assert demo.json()["provenance"] == "live"


def test_unsupported_task_is_explicit(tmp_path) -> None:
    service, bundle, _runtime = make_service(tmp_path)
    artifact = bundle.artifacts_by_model["model_sequence_kd"]
    narrowed = artifact.model_copy(update={"supported_tasks": ["transaction_review"]})
    bundle.artifacts_by_model["model_sequence_kd"] = narrowed
    bundle.artifacts_by_id[narrowed.artifact_id] = narrowed
    with pytest.raises(InferenceError) as exc_info:
        service.infer(
            InferRequest(
                model_id="model_sequence_kd",
                artifact_id="artifact_sequence_kd",
                task="cash_reconciliation",
                example_id=None,
                input=sample_input("cash_reconciliation"),
            )
        )
    assert exc_info.value.code == InferenceErrorCode.UNSUPPORTED_TASK
