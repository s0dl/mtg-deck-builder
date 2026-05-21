from __future__ import annotations

from typing import Any

from app.mcp.server import DeckBuilderMcpServer
from app.models.deck import DeckRequest, Format
from app.rag.retriever import RagRetriever
from app.skills.deck_evaluation import evaluate_candidate_card


class DeckAgentTools:
    """Agent-facing client facade over the deck-builder MCP server."""

    def __init__(
        self,
        mcp_server: DeckBuilderMcpServer | None = None,
        retriever: RagRetriever | None = None,
    ) -> None:
        if mcp_server is None:
            mcp_server = DeckBuilderMcpServer(retriever=retriever)
        self.mcp_server = mcp_server

    def list_tools(self) -> list[dict[str, Any]]:
        return self.mcp_server.list_tools()

    def tool_signatures(self, names: tuple[str, ...] | list[str]) -> list[str]:
        return self.mcp_server.tool_signatures(names)

    def search_strategy(
        self,
        query: str,
        mtg_format: Format | None = None,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        return self.mcp_server.call_tool_sync(
            "search_strategy",
            {"query": query, "mtg_format": _format_value(mtg_format), "limit": limit},
        )

    def search_meta_decks(
        self,
        query: str,
        mtg_format: Format | None = None,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        return self.mcp_server.call_tool_sync(
            "search_meta_decks",
            {"query": query, "mtg_format": _format_value(mtg_format), "limit": limit},
        )

    def search_rules(self, intent: str, limit: int = 20) -> list[dict[str, Any]]:
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

    def evaluate_deck_candidates(
        self,
        cards: list[dict[str, Any]],
        request: DeckRequest,
    ) -> list[dict[str, Any]]:
        evaluations: list[dict[str, Any]] = []
        for card in cards:
            if not isinstance(card, dict):
                continue
            evaluation = evaluate_candidate_card(card, request)
            evaluations.append(
                {
                    "name": evaluation.name,
                    "score": evaluation.score,
                    "is_playable": evaluation.is_playable,
                    "role": evaluation.role,
                    "reasons": list(evaluation.reasons),
                }
            )
        evaluations.sort(key=lambda item: item["score"], reverse=True)
        return evaluations

    def curate_context_notes(
        self,
        documents: list[dict[str, Any]],
        max_notes: int = 15,
    ) -> list[str]:
        notes: list[str] = []
        seen: set[str] = set()
        for document in documents:
            if not isinstance(document, dict):
                continue
            title = str(document.get("title") or "").strip()
            source = str(document.get("source") or "").strip()
            content = " ".join(str(document.get("content") or "").split())
            if not title or not content:
                continue
            key = f"{source}:{title}:{content[:80]}"
            if key in seen:
                continue
            seen.add(key)
            source_label = f" ({source})" if source else ""
            notes.append(f"{title}{source_label}: {content[:360]}")
            if len(notes) >= max_notes:
                break
        return notes


def _format_value(mtg_format: Format | None) -> str | None:
    return mtg_format.value if mtg_format is not None else None
