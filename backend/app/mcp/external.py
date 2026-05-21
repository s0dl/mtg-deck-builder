from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.api.decks import generate_deck
from app.core.database import SessionLocal
from app.mcp.server import DeckBuilderMcpServer
from app.models.deck import DeckRequest, Format
from app.rag.retriever import RagRetriever

ServerContextFactory = Callable[[], AbstractContextManager[DeckBuilderMcpServer]]


@contextmanager
def _default_server_context() -> Iterator[DeckBuilderMcpServer]:
    session = SessionLocal()
    try:
        yield DeckBuilderMcpServer(retriever=RagRetriever(session))
    finally:
        session.close()


def _call_tool(
    server_context_factory: ServerContextFactory,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    with server_context_factory() as server:
        return server.call_tool_sync(tool_name, arguments)


def create_deck_builder_mcp_app(
    server_context_factory: ServerContextFactory | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8010,
) -> FastMCP[None]:
    context_factory = server_context_factory or _default_server_context
    app = FastMCP("mtg-deck-builder", host=host, port=port)

    @app.tool(description="Search indexed RAG documents by text terms.")
    def search_rag_text(
        query: str,
        limit: int = 5,
        source: str | None = None,
        metadata_filters: dict[str, list[str]] | None = None,
    ) -> Any:
        return _call_tool(
            context_factory,
            "search_rag_text",
            {
                "query": query,
                "limit": limit,
                "source": source,
                "metadata_filters": metadata_filters,
            },
        )

    @app.tool(description="Search indexed RAG documents by semantic vector similarity.")
    def search_rag_vector(
        query: str,
        limit: int = 5,
        source: str | None = None,
    ) -> Any:
        return _call_tool(
            context_factory,
            "search_rag_vector",
            {
                "query": query,
                "limit": limit,
                "source": source,
            },
        )

    @app.tool(description="Search indexed RAG documents by metadata prefix.")
    def search_rag_metadata_prefixes(
        source: str,
        metadata_key: str,
        prefixes: list[str],
        limit: int = 10,
    ) -> Any:
        return _call_tool(
            context_factory,
            "search_rag_metadata_prefixes",
            {
                "source": source,
                "metadata_key": metadata_key,
                "prefixes": prefixes,
                "limit": limit,
            },
        )

    @app.tool(description="Search strategy articles and meta deck snapshots for a deck request.")
    def search_strategy(
        query: str,
        mtg_format: Format | None = None,
        limit: int = 40,
    ) -> Any:
        return _call_tool(
            context_factory,
            "search_strategy",
            {
                "query": query,
                "mtg_format": mtg_format,
                "limit": limit,
            },
        )

    @app.tool(description="Search MTGDecks archetype and top-deck snapshots.")
    def search_meta_decks(
        query: str,
        mtg_format: Format | None = None,
        limit: int = 40,
    ) -> Any:
        return _call_tool(
            context_factory,
            "search_meta_decks",
            {
                "query": query,
                "mtg_format": mtg_format,
                "limit": limit,
            },
        )

    @app.tool(description="Search Magic comprehensive rules RAG documents.")
    def search_rules(
        intent: str,
        limit: int = 20,
    ) -> Any:
        return _call_tool(
            context_factory,
            "search_rules",
            {
                "intent": intent,
                "limit": limit,
            },
        )

    @app.tool(description="Search the indexed Scryfall bulk card corpus with deck-request filters.")
    def search_card_corpus(
        query: str,
        mtg_format: Format | None = None,
        limit: int = 20,
        request: DeckRequest | None = None,
    ) -> Any:
        return _call_tool(
            context_factory,
            "search_card_corpus",
            {
                "query": query,
                "mtg_format": mtg_format,
                "limit": limit,
                "request": request,
            },
        )

    @app.tool(description="Search the indexed Scryfall card corpus for current card facts, prices, and legality.")
    def search_cards_scryfall(
        query: str,
        limit: int = 20,
    ) -> Any:
        return _call_tool(
            context_factory,
            "search_cards_scryfall",
            {
                "query": query,
                "limit": limit,
            },
        )

    @app.tool(description="Look up one exact card name in the indexed Scryfall card corpus.")
    def lookup_card(name: str) -> Any:
        return _call_tool(
            context_factory,
            "lookup_card",
            {
                "name": name,
            },
        )

    @app.tool(description="Validate deck size and copy limits for a format.")
    def validate_deck_cards(cards: list[dict[str, Any]], mtg_format: Format) -> Any:
        return _call_tool(
            context_factory,
            "validate_deck_cards",
            {
                "cards": cards,
                "mtg_format": mtg_format,
            },
        )

    @app.tool(description="Build a deck using the same request fields as the web app. (format, budget, colors, playstyle, strategy, must-include, avoid)")
    async def build_deck(
        format: Format = Format.modern,
        budget_usd: float | None = None,
        colors: list[str] | None = None,
        playstyle: str = "",
        strategy: str = "",
        must_include: list[str] | None = None,
        avoid: list[str] | None = None,
    ) -> Any:
        session = SessionLocal()
        try:
            response = await generate_deck(
                DeckRequest(
                    format=format,
                    budget_usd=budget_usd,
                    colors=list(colors or []),
                    playstyle=playstyle,
                    strategy=strategy,
                    must_include=list(must_include or []),
                    avoid=list(avoid or []),
                ),
                session=session,
            )
            return response.model_dump(mode="json")
        finally:
            session.close()

    return app


def _mcp_host() -> str:
    return os.getenv("MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _mcp_port() -> int:
    return int(os.getenv("MCP_PORT", "8010"))


mcp = create_deck_builder_mcp_app(host=_mcp_host(), port=_mcp_port())


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip() or "stdio"
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
