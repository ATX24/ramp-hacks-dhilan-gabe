"""SageMaker-compatible inference HTTP API: /ping, /invocations, demo routes."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from distillery_inference.errors import InferenceError, InferenceErrorCode
from distillery_inference.schemas import InferRequest
from distillery_inference.service import InferenceService

_SERVICE: InferenceService | None = None


def get_service() -> InferenceService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = InferenceService.create()
    return _SERVICE


def create_app(service: InferenceService | None = None) -> FastAPI:
    app = FastAPI(
        title="Distillery Inference",
        version="0.1.0",
        description=(
            "SageMaker real-time inference plane for Demo/Playground. "
            "Loads sealed local model bundles only; no fabricated responses."
        ),
    )
    if service is not None:
        app.state.service = service
    else:
        app.state.service = None

    @app.exception_handler(InferenceError)
    async def inference_error_handler(
        _request: Request,
        exc: InferenceError,
    ) -> JSONResponse:
        payload = exc.to_payload()
        return JSONResponse(status_code=exc.http_status, content=payload)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "provenance": "none",
                "code": InferenceErrorCode.INTERNAL_ERROR.value,
                "message": "Request validation failed.",
                "retryable": False,
                "details": {"errors": exc.errors()},
            },
        )

    def resolve_service(request: Request) -> InferenceService:
        existing = getattr(request.app.state, "service", None)
        if existing is not None:
            return existing
        created = get_service()
        request.app.state.service = created
        return created

    @app.get("/ping")
    async def ping(request: Request) -> PlainTextResponse:
        service_obj = resolve_service(request)
        if not service_obj.ready:
            return PlainTextResponse("not ready", status_code=503)
        return PlainTextResponse("ok", status_code=200)

    @app.get("/health")
    @app.get("/v1/demo/health")
    async def health(request: Request) -> dict[str, Any]:
        return resolve_service(request).health().model_dump(mode="json")

    @app.get("/v1/demo/models")
    async def models(request: Request) -> dict[str, Any]:
        return resolve_service(request).models().model_dump(mode="json")

    @app.post("/invocations")
    @app.post("/v1/demo/infer")
    async def invoke(request: Request) -> dict[str, Any]:
        service_obj = resolve_service(request)
        body = await request.json()
        if not isinstance(body, dict):
            raise InferenceError(
                InferenceErrorCode.INTERNAL_ERROR,
                "Invocation body must be a JSON object.",
                http_status=400,
            )
        infer_request = InferRequest.model_validate(body)
        result = service_obj.infer(infer_request)
        return result.model_dump(mode="json")

    return app


app = create_app()
