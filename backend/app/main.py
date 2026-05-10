from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.decks import router as decks_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="MTG Deck Builder Agent",
        version="0.1.0",
        description="MCP, RAG, and deterministic skills for Magic deck construction.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(decks_router, prefix="/api")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
