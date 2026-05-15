from __future__ import annotations

from typing import Any

from app.mcp.scryfall_client import ScryfallClient
from app.mcp.server import DeckBuilderMcpServer
from app.models.deck import DeckRequest, Format
from app.rag.retriever import RagRetriever


class DeckAgentTools:
    """Agent-facing client facade over the deck-builder MCP server."""

    def __init__(
        self,
        mcp_server: DeckBuilderMcpServer | None = None,
        retriever: RagRetriever | None = None,
        scryfall: ScryfallClient | None = None,
    ) -> None:
        if mcp_server is None:
            mcp_server = DeckBuilderMcpServer(retriever=retriever, scryfall=scryfall)
        self.mcp_server = mcp_server

    def list_tools(self) -> list[dict[str, Any]]:
        return self.mcp_server.list_tools()

    def tool_signatures(self, names: tuple[str, ...] | list[str]) -> list[str]:
        return self.mcp_server.tool_signatures(names)

    def search_strategy(
        self,
        query: str,
        mtg_format: Format | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        return self.mcp_server.call_tool_sync(
            "search_strategy",
            {"query": query, "mtg_format": _format_value(mtg_format), "limit": limit},
        )

    def search_meta_decks(
        self,
        query: str,
        mtg_format: Format | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        return self.mcp_server.call_tool_sync(
            "search_meta_decks",
            {"query": query, "mtg_format": _format_value(mtg_format), "limit": limit},
        )

    def search_rules(self, intent: str, limit: int = 6) -> list[dict[str, Any]]:
        return self.mcp_server.call_tool_sync(
            "search_rules",
            {"intent": intent, "limit": limit},
        )

    def search_card_corpus(
        self,
        query: str,
        mtg_format: Format | None = None,
        limit: int = 20,
        request: DeckRequest | None = None,
    ) -> list[dict[str, Any]]:
        return self.mcp_server.call_tool_sync(
            "search_card_corpus",
            {
                "query": query,
                "mtg_format": _format_value(mtg_format),
                "limit": limit,
                "request": request.model_dump(mode="json") if request is not None else None,
            },
        )

    async def search_cards_scryfall(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return await self.mcp_server.call_tool(
            "search_cards_scryfall",
            {"query": query, "limit": limit},
        )

    async def lookup_card(self, name: str) -> dict[str, Any]:
        return await self.mcp_server.call_tool("lookup_card", {"name": name})

    def validate_deck_cards(
        self,
        cards: list[dict[str, Any]],
        mtg_format: Format,
    ) -> dict[str, Any]:
        return self.mcp_server.call_tool_sync(
            "validate_deck_cards",
            {"cards": cards, "mtg_format": mtg_format.value},
        )


def _format_value(mtg_format: Format | None) -> str | None:
    return mtg_format.value if mtg_format is not None else None
