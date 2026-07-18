"""Serving orchestration: registry lookup, limits, validation, adapter switching."""

from __future__ import annotations

import json
import threading
import time

from distillery_inference.bundle import LoadedBundle, load_serving_bundle
from distillery_inference.config import InferenceSettings, enforce_offline_environment
from distillery_inference.errors import InferenceError, InferenceErrorCode
from distillery_inference.runtime import InferenceRuntime, build_runtime
from distillery_inference.schemas import (
    HealthResponse,
    InferOkResponse,
    InferRequest,
    ModelRegistryResponse,
)
from distillery_inference.stats import build_model_registry_response
from distillery_inference.validation import parse_json_object, validate_structured_output


class InferenceService:
    """Concurrency-safe inference service with serialized adapter switches."""

    def __init__(
        self,
        *,
        settings: InferenceSettings,
        bundle: LoadedBundle,
        runtime: InferenceRuntime,
    ) -> None:
        self.settings = settings
        self.bundle = bundle
        self.runtime = runtime
        self._lock = threading.RLock()
        self._inflight = 0
        self._ready = runtime.ready()

    @classmethod
    def create(
        cls,
        settings: InferenceSettings | None = None,
        *,
        runtime: InferenceRuntime | None = None,
        skip_offline_check: bool = False,
    ) -> InferenceService:
        resolved = settings or InferenceSettings.from_environ()
        if resolved.require_offline and not skip_offline_check:
            enforce_offline_environment()
        bundle = load_serving_bundle(resolved.model_bundle_root)
        resolved_runtime = runtime or build_runtime(
            bundle=bundle,
            backend=resolved.runtime_backend,
        )
        return cls(settings=resolved, bundle=bundle, runtime=resolved_runtime)

    @property
    def ready(self) -> bool:
        return self._ready and self.runtime.ready()

    def health(self) -> HealthResponse:
        return HealthResponse(
            serving_ready=self.ready,
            endpoint_id=self.settings.endpoint_id,
            available_model_ids=self.runtime.available_model_ids(),
            loaded_model_id=self.runtime.loaded_model_id(),
            offline_enforced=self.settings.require_offline,
        )

    def models(self) -> ModelRegistryResponse:
        return build_model_registry_response(
            self.bundle,
            endpoint_id=self.settings.endpoint_id,
        )

    def infer(self, request: InferRequest) -> InferOkResponse:
        if not self.ready:
            raise InferenceError(
                InferenceErrorCode.SERVING_NOT_READY,
                "Inference serving is not ready.",
                http_status=503,
                retryable=True,
            )
        payload_bytes = len(
            json.dumps(request.input, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        if payload_bytes > self.settings.max_input_bytes:
            raise InferenceError(
                InferenceErrorCode.INPUT_TOO_LARGE,
                (
                    f"Input payload {payload_bytes} bytes exceeds limit "
                    f"{self.settings.max_input_bytes}"
                ),
                http_status=413,
            )

        artifact = self.bundle.resolve_artifact(
            model_id=request.model_id,
            artifact_id=request.artifact_id,
        )
        if request.task not in artifact.supported_tasks:
            raise InferenceError(
                InferenceErrorCode.UNSUPPORTED_TASK,
                (f"Task {request.task} is not supported by artifact {artifact.artifact_id}"),
                http_status=400,
                details={"supported_tasks": list(artifact.supported_tasks)},
            )

        started = time.perf_counter()
        with self._lock:
            if self._inflight >= self.settings.max_concurrent_requests:
                raise InferenceError(
                    InferenceErrorCode.REQUEST_LIMIT_EXCEEDED,
                    "Too many concurrent inference requests.",
                    http_status=429,
                    retryable=True,
                )
            self._inflight += 1
            try:
                # Serialize adapter switch + generate for demo QPS safety.
                self.runtime.ensure_artifact(artifact)
                if self.runtime.loaded_model_id() != artifact.model_id:
                    raise InferenceError(
                        InferenceErrorCode.ADAPTER_SWITCH_FAILED,
                        (
                            "Loaded model does not match requested model; "
                            "refusing silent substitution."
                        ),
                        http_status=500,
                        details={
                            "requested_model_id": artifact.model_id,
                            "loaded_model_id": self.runtime.loaded_model_id(),
                        },
                    )
                generation = self.runtime.generate(
                    artifact=artifact,
                    task=request.task,
                    example_input=request.input,
                    max_prompt_tokens=self.settings.max_prompt_tokens,
                    max_completion_tokens=self.settings.max_completion_tokens,
                    temperature=self.settings.temperature,
                    top_p=self.settings.top_p,
                    seed=self.settings.seed,
                    timeout_s=self.settings.request_timeout_s,
                )
            finally:
                self._inflight -= 1

        if generation.model_id != request.model_id:
            raise InferenceError(
                InferenceErrorCode.ADAPTER_SWITCH_FAILED,
                "Runtime returned a different model_id; refusing silent substitution.",
                http_status=500,
            )
        if generation.artifact_id != request.artifact_id:
            raise InferenceError(
                InferenceErrorCode.ARTIFACT_ID_MISMATCH,
                "Runtime returned a different artifact_id; refusing silent substitution.",
                http_status=500,
            )

        try:
            structured = parse_json_object(generation.raw_output)
        except ValueError as exc:
            raise InferenceError(
                InferenceErrorCode.MALFORMED_OUTPUT,
                f"Model output was not valid structured JSON: {exc}",
                http_status=502,
                details={"raw_output": generation.raw_output[:2000]},
            ) from exc

        validation, detail = validate_structured_output(request.task, structured)
        if validation != "valid":
            raise InferenceError(
                InferenceErrorCode.STRUCTURED_OUTPUT_INVALID,
                detail or "Structured output failed task schema validation.",
                http_status=502,
                details={"structured_output": structured},
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        return InferOkResponse(
            model_id=request.model_id,
            artifact_id=request.artifact_id,
            task=request.task,
            example_id=request.example_id,
            structured_output=structured,
            raw_output=generation.raw_output,
            validation=validation,
            validation_detail=detail,
            latency_ms=latency_ms,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
        )
