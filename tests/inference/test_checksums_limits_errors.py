"""Checksum failures, malformed outputs, token limits, timeouts, offline mode."""

from __future__ import annotations

import json
import socket

import pytest
from distillery_inference.bundle import load_serving_bundle
from distillery_inference.config import enforce_offline_environment
from distillery_inference.errors import InferenceError, InferenceErrorCode
from distillery_inference.runtime import FakeRuntime
from distillery_inference.schemas import InferRequest
from distillery_inference.server import create_app
from distillery_inference.service import InferenceService
from fastapi.testclient import TestClient
from support import build_bundle, make_service, make_settings, sample_input


def test_checksum_mismatch_blocks_ready(tmp_path) -> None:
    root = build_bundle(
        tmp_path / "bundle",
        corrupt_checksum_for="adapters/sequence_kd/adapter_model.safetensors",
    )
    with pytest.raises(InferenceError) as exc_info:
        load_serving_bundle(root)
    assert exc_info.value.code == InferenceErrorCode.ARTIFACT_CHECKSUM_FAILED


def test_malformed_output_maps_to_explicit_error(tmp_path) -> None:
    service, bundle, runtime = make_service(tmp_path)
    runtime.outputs_by_model["model_sequence_kd"] = "not-json-at-all"
    with pytest.raises(InferenceError) as exc_info:
        service.infer(
            InferRequest(
                model_id="model_sequence_kd",
                artifact_id=bundle.artifacts_by_model["model_sequence_kd"].artifact_id,
                task="transaction_review",
                example_id=None,
                input=sample_input(),
            )
        )
    assert exc_info.value.code == InferenceErrorCode.MALFORMED_OUTPUT


def test_structured_output_schema_failure(tmp_path) -> None:
    service, bundle, runtime = make_service(tmp_path)
    runtime.outputs_by_model["model_sequence_kd"] = json.dumps(
        {"task": "transaction_review", "schema_version": "transaction_review.v1"}
    )
    with pytest.raises(InferenceError) as exc_info:
        service.infer(
            InferRequest(
                model_id="model_sequence_kd",
                artifact_id=bundle.artifacts_by_model["model_sequence_kd"].artifact_id,
                task="transaction_review",
                example_id=None,
                input=sample_input(),
            )
        )
    assert exc_info.value.code == InferenceErrorCode.STRUCTURED_OUTPUT_INVALID


def test_token_limit_exceeded(tmp_path) -> None:
    service, bundle, runtime = make_service(
        tmp_path,
        settings_overrides={"max_prompt_tokens": 8},
    )
    runtime.token_overflow_models.add("model_sequence_kd")
    with pytest.raises(InferenceError) as exc_info:
        service.infer(
            InferRequest(
                model_id="model_sequence_kd",
                artifact_id=bundle.artifacts_by_model["model_sequence_kd"].artifact_id,
                task="transaction_review",
                example_id=None,
                input=sample_input(),
            )
        )
    assert exc_info.value.code == InferenceErrorCode.TOKEN_LIMIT_EXCEEDED


def test_timeout_error_mapping(tmp_path) -> None:
    service, bundle, runtime = make_service(
        tmp_path,
        settings_overrides={"request_timeout_s": 0.05},
    )
    runtime.timeout_models.add("model_sequence_kd")
    with pytest.raises(InferenceError) as exc_info:
        service.infer(
            InferRequest(
                model_id="model_sequence_kd",
                artifact_id=bundle.artifacts_by_model["model_sequence_kd"].artifact_id,
                task="transaction_review",
                example_id=None,
                input=sample_input(),
            )
        )
    assert exc_info.value.code == InferenceErrorCode.TIMEOUT
    assert exc_info.value.retryable is True


def test_http_error_mapping(tmp_path) -> None:
    service, bundle, runtime = make_service(tmp_path)
    runtime.outputs_by_model["model_sequence_kd"] = "@@@broken"
    client = TestClient(create_app(service=service))
    response = client.post(
        "/invocations",
        json={
            "model_id": "model_sequence_kd",
            "artifact_id": bundle.artifacts_by_model["model_sequence_kd"].artifact_id,
            "task": "transaction_review",
            "example_id": None,
            "input": sample_input(),
        },
    )
    assert response.status_code == 502
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["provenance"] == "none"
    assert payload["code"] == InferenceErrorCode.MALFORMED_OUTPUT.value


def test_offline_environment_required() -> None:
    with pytest.raises(RuntimeError):
        enforce_offline_environment(
            {
                "HF_HUB_OFFLINE": "0",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )


def test_no_network_during_fake_inference(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def blocked(*args, **kwargs):  # noqa: ANN001
        calls.append((args, kwargs))
        raise AssertionError("network access attempted during fake inference")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)

    service, bundle, _runtime = make_service(tmp_path)
    service.infer(
        InferRequest(
            model_id="model_logit_kd",
            artifact_id=bundle.artifacts_by_model["model_logit_kd"].artifact_id,
            task="variance_analysis",
            example_id=None,
            input=sample_input("variance_analysis"),
        )
    )
    assert calls == []


def test_input_too_large(tmp_path) -> None:
    service, bundle, _runtime = make_service(
        tmp_path,
        settings_overrides={"max_input_bytes": 64},
    )
    with pytest.raises(InferenceError) as exc_info:
        service.infer(
            InferRequest(
                model_id="model_sequence_kd",
                artifact_id=bundle.artifacts_by_model["model_sequence_kd"].artifact_id,
                task="transaction_review",
                example_id=None,
                input={"blob": "x" * 200},
            )
        )
    assert exc_info.value.code == InferenceErrorCode.INPUT_TOO_LARGE


def test_stats_unknown_when_evidence_absent(tmp_path) -> None:
    bundle_root = build_bundle(tmp_path / "bundle")
    bundle = load_serving_bundle(bundle_root)
    # Clear stats evidence for one artifact.
    artifact = bundle.artifacts_by_model["model_student_base"]
    object.__setattr__(artifact, "stats", {})
    object.__setattr__(artifact, "proof_status", None)
    object.__setattr__(artifact, "recipe", None)
    settings = make_settings(bundle_root)
    service = InferenceService(
        settings=settings,
        bundle=bundle,
        runtime=FakeRuntime(bundle=bundle),
    )
    registry = service.models()
    base = next(model for model in registry.models if model.model_id == "model_student_base")
    assert base.stats.proof_status is None
    assert base.stats.recipe is None
    assert base.stats.promotion_status == "unknown"
