from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.alignment import AlignmentModelError, AlignmentProcessingError, AlignmentService
from app.config import Settings
from app.languages import SUPPORTED_LANGUAGES
from app.llm_alignment import LLMAlignmentService
from app.llm_client import LLMProviderError
from app.llm_config import LLMSettings
from app.llm_schemas import LLMAlignmentRequest, LLMAlignmentResponse, LLMProviderInfo, LLMProvidersResponse
from app.schemas import AlignmentRequest, AlignmentResponse, HealthResponse, SupportedLanguagesResponse

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    alignment_service: AlignmentService | None = None,
    llm_settings: LLMSettings | None = None,
    llm_alignment_service: LLMAlignmentService | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    service = alignment_service or AlignmentService(resolved_settings)
    llm_service = llm_alignment_service or LLMAlignmentService(llm_settings or LLMSettings.from_env())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if resolved_settings.eager_load:
            await run_in_threadpool(service.warm_up)
        yield

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description=(
            "Align translated sentence pairs with contextual token embeddings or configured "
            "OpenAI-compatible LLM providers. Embedding links include estimated scores; "
            "LLM alignments return validated groups without calibrated scores."
        ),
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.alignment_service = service
    application.state.llm_alignment_service = llm_service

    @application.exception_handler(LLMProviderError)
    async def handle_llm_error(_: Request, exc: LLMProviderError) -> JSONResponse:
        # Provider bodies and credentials never belong in public errors or logs.
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "message": str(exc)})

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

    @application.get("/demo", include_in_schema=False)
    async def demo() -> FileResponse:
        demo_path = Path(__file__).resolve().parent / "static" / "align.html"
        return FileResponse(demo_path, media_type="text/html")

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

    @application.get(
        "/api/v1/llm/providers",
        response_model=LLMProvidersResponse,
        tags=["llm-alignment"],
        summary="List configured OpenAI-compatible alignment providers",
    )
    async def llm_providers() -> LLMProvidersResponse:
        return LLMProvidersResponse(
            providers=[
                LLMProviderInfo(id=provider.id, name=provider.name, model=provider.model)
                for provider in llm_service.settings.providers
            ],
            default_provider=llm_service.settings.default_provider,
        )

    @application.post(
        "/api/v1/llm/align",
        response_model=LLMAlignmentResponse,
        tags=["llm-alignment"],
        summary="Align translated passages directly with a configured LLM",
        description=(
            "The LLM returns unit IDs; the server validates complete, non-overlapping ID coverage. "
            "Token offsets use UTF-16 code units in the unchanged input. "
            "Similarity and confidence are null because they are not calibrated. "
            "Requests allow 1–10 pairs, 1,024 units per side, and 4,096 units in total."
        ),
        responses={
            400: {"description": "Unknown provider"},
            422: {"description": "Invalid input or input exceeds alignment unit limits"},
            429: {"description": "Provider rate limit reached"},
            502: {"description": "Provider failure or invalid model alignment output"},
            503: {"description": "No LLM provider configured"},
            504: {"description": "Provider timeout"},
        },
    )
    async def llm_align(request: LLMAlignmentRequest) -> LLMAlignmentResponse:
        return await llm_service.align(request)

    return application


app = create_app()
