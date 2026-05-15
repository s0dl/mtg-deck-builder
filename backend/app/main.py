import logging
import time
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from app.api.agent import router as agent_router
from app.api.decks import router as decks_router
from app.core.config import get_settings
from app.core.logging import bind_request_id, configure_logging, log_extra, reset_request_id

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger.info(
        "Starting MTG Deck Builder backend",
        extra=log_extra(
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            openai_enabled=settings.openai_enabled,
            agent_provider=settings.agent_provider,
            ollama_model=settings.ollama_model,
            cors_origins=settings.cors_origins,
        ),
    )
    app = FastAPI(
        title="MTG Deck Builder Agent",
        version="0.1.0",
        description="MCP, RAG, and deterministic skills for Magic deck construction.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials="*" not in settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(decks_router, prefix="/api")
    app.include_router(agent_router, prefix="/api")

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        token = bind_request_id(request_id)
        start = time.perf_counter()
        logger.info(
            "HTTP request started",
            extra=log_extra(method=request.method, path=request.url.path),
        )
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.exception(
                "HTTP request failed",
                extra=log_extra(method=request.method, path=request.url.path, duration_ms=duration_ms),
            )
            raise
        finally:
            reset_request_id(token)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["x-request-id"] = request_id
        logger.info(
            "HTTP request completed",
            extra=log_extra(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            ),
        )
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
