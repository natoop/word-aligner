from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.alignment import AlignmentModelError, AlignmentProcessingError, AlignmentService
from app.config import Settings
from app.languages import SUPPORTED_LANGUAGES
from app.schemas import AlignmentRequest, AlignmentResponse, HealthResponse, SupportedLanguagesResponse

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    alignment_service: AlignmentService | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    service = alignment_service or AlignmentService(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if resolved_settings.eager_load:
            await run_in_threadpool(service.warm_up)
        yield

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description=(
            "Align words in translated sentence pairs with SimAlign. "
            "Alignment groups explicitly represent one-to-many and many-to-one relations."
        ),
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.alignment_service = service

    @application.exception_handler(AlignmentModelError)
    async def handle_model_error(_: Request, exc: AlignmentModelError) -> JSONResponse:
        logger.exception("Alignment model is unavailable", exc_info=exc)
        return JSONResponse(
            status_code=503,
            content={
                "error": "alignment_model_unavailable",
                "message": str(exc),
            },
        )

    @application.exception_handler(AlignmentProcessingError)
    async def handle_processing_error(_: Request, exc: AlignmentProcessingError) -> JSONResponse:
        logger.exception("Word alignment failed", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "alignment_failed",
                "message": str(exc),
            },
        )

    @application.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": resolved_settings.app_name,
            "version": resolved_settings.app_version,
            "docs": "/docs",
        }

    @application.get(
        "/health/live",
        response_model=HealthResponse,
        response_model_exclude_none=True,
        tags=["health"],
    )
    async def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get(
        "/health/ready",
        response_model=HealthResponse,
        response_model_exclude_none=True,
        tags=["health"],
    )
    async def ready() -> HealthResponse:
        return HealthResponse(
            status="ready",
            model=resolved_settings.model,
            model_loaded=service.model_loaded,
            load_mode="eager" if resolved_settings.eager_load else "lazy",
        )

    @application.get(
        "/api/v1/languages",
        response_model=SupportedLanguagesResponse,
        tags=["alignment"],
        summary="List supported word-alignment languages",
    )
    async def supported_languages() -> SupportedLanguagesResponse:
        return SupportedLanguagesResponse(
            model=resolved_settings.model,
            total=len(SUPPORTED_LANGUAGES),
            languages=list(SUPPORTED_LANGUAGES),
        )

    @application.post(
        "/api/v1/align",
        response_model=AlignmentResponse,
        tags=["alignment"],
        summary="Align translated sentence pairs",
    )
    async def align(request: AlignmentRequest) -> AlignmentResponse:
        return await run_in_threadpool(service.align, request)

    return application


app = create_app()
