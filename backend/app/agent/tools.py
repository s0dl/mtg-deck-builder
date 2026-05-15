from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.mcp.scryfall_client import ScryfallClient, scryfall_card_to_document
from app.models.deck import Format
from app.models.deck import DeckRequest
from app.rag.retriever import RagRetriever, RetrievedDocument
from app.skills.deck_rules import validate_deck
from app.skills.deck_evaluation import evaluate_candidate_document

NON_DECK_TYPE_MARKERS = {
    "Attraction",
    "Card",
    "Conspiracy",
    "Dungeon",
    "Emblem",
    "Phenomenon",
    "Plane",
    "Scheme",
    "Sticker",
    "Token",
    "Vanguard",
}

LAND_QUERY_TERMS = {
    "land",
    "lands",
    "dual",
    "fetch",
    "shock",
    "pain",
    "fast",
    "slow",
    "check",
    "pathway",
    "triome",
    "mana",
    "fixing",
}

GENERIC_QUERY_TERMS = {
    "and",
    "the",
    "for",
    "with",
    "card",
    "cards",
    "format",
    "legal",
    "modern",
    "standard",
    "pioneer",
    "legacy",
    "vintage",
    "commander",
    "pauper",
    "casual",
    "blue",
    "red",
    "white",
    "black",
    "green",
    "color",
    "identity",
}


def _document_payload(document: RetrievedDocument) -> dict[str, Any]:
    return {
        "title": document.title,
        "content": document.content,
        "source": document.source,
        "metadata": document.metadata,
        "score": document.score,
    }


@dataclass
class DeckAgentTools:
    """Constrained tools deck-building agents are allowed to call."""

    retriever: RagRetriever
    scryfall: ScryfallClient

    def search_strategy(
        self,
        query: str,
        mtg_format: Format | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        article_limit = max(1, limit // 2)
        meta_limit = max(1, limit - article_limit)
        documents = [
            *self._search_strategy_source(
                query=query,
                source="mtgdecks_articles",
                mtg_format=mtg_format,
                limit=article_limit,
            ),
            *self._search_strategy_source(
                query=query,
                source="mtgdecks_meta_decks",
                mtg_format=mtg_format,
                limit=meta_limit,
            ),
        ]
        return [_document_payload(document) for document in _dedupe_documents(documents)[:limit]]

    def search_meta_decks(
        self,
        query: str,
        mtg_format: Format | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        return [
            _document_payload(document)
            for document in self._search_strategy_source(
                query=query,
                source="mtgdecks_meta_decks",
                mtg_format=mtg_format,
                limit=limit,
            )
        ]

    def _search_strategy_source(
        self,
        query: str,
        source: str,
        mtg_format: Format | None,
        limit: int,
    ) -> list[RetrievedDocument]:
        metadata_filters = None
        if mtg_format is not None and mtg_format != Format.casual:
            metadata_filters = {"format": [mtg_format.value, ""]}
        return self.retriever.search_text(
            query=query,
            limit=limit,
            source=source,
            metadata_filters=metadata_filters,
        )

    def search_rules(self, intent: str, limit: int = 6) -> list[dict[str, Any]]:
        return [
            _document_payload(document)
            for document in self.retriever.search_text(
                query=intent,
                limit=limit,
                source="mtg_comprehensive_rules",
            )
        ]

    def search_card_corpus(
        self,
        query: str,
        mtg_format: Format | None = None,
        limit: int = 20,
        request: DeckRequest | None = None,
    ) -> list[dict[str, Any]]:
        search_limit = max(limit * 6, 80)
        documents = _dedupe_documents(
            [
                *self.retriever.search(query=query, limit=search_limit, source="scryfall_bulk"),
                *self.retriever.search_text(query=query, limit=search_limit, source="scryfall_bulk"),
            ]
        )
        ranked: list[tuple[float, RetrievedDocument]] = []
        for document in documents:
            if not _is_allowed_card_document(document, query=query, mtg_format=mtg_format, request=request):
                continue
            ranked.append((_card_relevance_score(document, query=query, request=request), document))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [_document_payload(document) for _, document in ranked[:limit]]

    async def search_cards_scryfall(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        payload = await self.scryfall.search_cards(query=query)
        documents = [scryfall_card_to_document(card) for card in payload.get("data", [])[:limit]]
        return [_document_payload(document) for document in documents]

    async def lookup_card(self, name: str) -> dict[str, Any]:
        return _document_payload(scryfall_card_to_document(await self.scryfall.get_card_named(name)))

    def validate_deck_cards(
        self,
        cards: list[dict[str, Any]],
        mtg_format: Format,
    ) -> dict[str, Any]:
        return validate_deck(cards=cards, mtg_format=mtg_format).model_dump()


def _is_allowed_card_document(
    document: RetrievedDocument,
    query: str,
    mtg_format: Format | None,
    request: DeckRequest | None,
) -> bool:
    type_line = str(document.metadata.get("type_line") or "")
    if any(marker in type_line.split(" ") for marker in NON_DECK_TYPE_MARKERS):
        return False

    if mtg_format is not None and mtg_format != Format.casual:
        legalities = document.metadata.get("legalities") or {}
        if legalities.get(mtg_format.value) != "legal":
            return False

    requested_colors = {color.upper() for color in (request.colors if request else [])}
    requested_colors = {color for color in requested_colors if color in {"W", "U", "B", "R", "G"}}
    if requested_colors:
        identity = {str(color).upper() for color in document.metadata.get("color_identity") or []}
        colors = {str(color).upper() for color in document.metadata.get("colors") or []}
        card_colors = identity or colors
        if not card_colors.issubset(requested_colors):
            return False

    haystack = _document_text(document)
    for avoid in request.avoid if request else []:
        if str(avoid).strip().lower() in haystack:
            return False

    is_land = "Land" in type_line.split(" ")
    if _is_land_query(query):
        return is_land
    return not is_land


def _card_relevance_score(document: RetrievedDocument, query: str, request: DeckRequest | None) -> float:
    score = float(document.score or 0.0)
    title = document.title.lower()
    type_line = str(document.metadata.get("type_line") or "").lower()
    content = str(document.content or "").lower()
    metadata_text = " ".join(str(value).lower() for value in document.metadata.values() if value)
    text = " ".join([title, type_line, content, metadata_text])

    for term in _important_terms(query):
        if term in title:
            score += 5.0
        if term in type_line:
            score += 3.0
        if term in metadata_text:
            score += 2.0
        if term in content:
            score += 1.0

    if request is not None:
        score += evaluate_candidate_document(document, request).score
        request_terms = _important_terms(" ".join([request.playstyle, request.strategy, " ".join(request.must_include)]))
        for term in request_terms:
            if term in text:
                score += 2.0
        if request.budget_usd is not None:
            price = document.metadata.get("estimated_price_usd")
            if isinstance(price, int | float):
                score += max(0.0, 3.0 - min(float(price), 30.0) / 10.0)

    return score


def _important_terms(query: str) -> list[str]:
    terms: list[str] = []
    for raw_term in query.lower().replace(":", " ").replace("<=", " ").replace("-", " ").split():
        term = raw_term.strip("()[]{}.,;\"'")
        if len(term) <= 2 or term in GENERIC_QUERY_TERMS:
            continue
        if term not in terms:
            terms.append(term)
    return terms


def _is_land_query(query: str) -> bool:
    return any(term in LAND_QUERY_TERMS for term in _important_terms(query))


def _document_text(document: RetrievedDocument) -> str:
    return " ".join(
        [
            document.title,
            document.content,
            str(document.metadata.get("name") or ""),
            str(document.metadata.get("type_line") or ""),
            " ".join(str(value) for value in document.metadata.values() if value),
        ]
    ).lower()


def _dedupe_documents(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    deduped: list[RetrievedDocument] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        name = str(document.metadata.get("name") or document.metadata.get("archetype") or document.title)
        key = (document.source, name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(document)
    return deduped
