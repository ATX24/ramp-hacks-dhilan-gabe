"""Registry-driven adapter switching and concurrency safety."""

from __future__ import annotations

import threading

import pytest
from distillery_inference.errors import InferenceError, InferenceErrorCode
from distillery_inference.schemas import InferRequest
from support import make_service, sample_input


def test_switch_among_registry_arms_without_hardcoded_branches(tmp_path) -> None:
    service, bundle, runtime = make_service(tmp_path)
    order = [
        "model_student_base",
        "model_oracle_sft",
        "model_sequence_kd",
        "model_logit_kd",
        "model_ce_ablation",
        "model_promoted_winner",
    ]
    for model_id in order:
        artifact = bundle.artifacts_by_model[model_id]
        result = service.infer(
            InferRequest(
                model_id=model_id,
                artifact_id=artifact.artifact_id,
                task="transaction_review",
                example_id=None,
                input=sample_input(),
            )
        )
        assert result.model_id == model_id
        assert result.artifact_id == artifact.artifact_id
        assert runtime.loaded_model_id() == model_id
    assert runtime.switch_count >= len(order)


def test_missing_model_and_artifact_mismatch_are_explicit(tmp_path) -> None:
    service, _bundle, _runtime = make_service(tmp_path)
    with pytest.raises(InferenceError) as missing:
        service.infer(
            InferRequest(
                model_id="model_does_not_exist",
                artifact_id="artifact_sequence_kd",
                task="transaction_review",
                example_id=None,
                input=sample_input(),
            )
        )
    assert missing.value.code == InferenceErrorCode.MODEL_NOT_IN_REGISTRY

    with pytest.raises(InferenceError) as mismatch:
        service.infer(
            InferRequest(
                model_id="model_sequence_kd",
                artifact_id="artifact_oracle_sft",
                task="transaction_review",
                example_id=None,
                input=sample_input(),
            )
        )
    assert mismatch.value.code == InferenceErrorCode.ARTIFACT_ID_MISMATCH


def test_excluded_artifact_is_unservable(tmp_path) -> None:
    service, _bundle, _runtime = make_service(
        tmp_path,
        exclude_arms={"ce_ablation"},
    )
    with pytest.raises(InferenceError) as exc_info:
        service.infer(
            InferRequest(
                model_id="model_ce_ablation",
                artifact_id="artifact_ce_ablation",
                task="transaction_review",
                example_id=None,
                input=sample_input(),
            )
        )
    assert exc_info.value.code == InferenceErrorCode.ARTIFACT_NOT_SERVABLE


def test_concurrent_switches_serialize_without_cross_model_leak(tmp_path) -> None:
    service, bundle, runtime = make_service(
        tmp_path,
        settings_overrides={"max_concurrent_requests": 4, "request_timeout_s": 5.0},
    )
    errors: list[BaseException] = []
    results: list[str] = []
    barrier = threading.Barrier(4)

    def worker(model_id: str) -> None:
        try:
            barrier.wait(timeout=2)
            artifact = bundle.artifacts_by_model[model_id]
            response = service.infer(
                InferRequest(
                    model_id=model_id,
                    artifact_id=artifact.artifact_id,
                    task="transaction_review",
                    example_id=None,
                    input=sample_input(),
                )
            )
            results.append(response.model_id)
            assert response.model_id == model_id
        except BaseException as exc:  # noqa: BLE001 - collect for assertion
            errors.append(exc)

    models = [
        "model_oracle_sft",
        "model_sequence_kd",
        "model_logit_kd",
        "model_promoted_winner",
    ]
    threads = [threading.Thread(target=worker, args=(model_id,)) for model_id in models]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert errors == []
    assert sorted(results) == sorted(models)
    assert runtime.loaded_model_id() in models
