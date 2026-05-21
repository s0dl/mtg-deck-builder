from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any

from app.models.deck import DeckRequest, Format
from app.rag.retriever import RagRetriever, RetrievedDocument
from app.skills.deck_evaluation import evaluate_candidate_document
from app.skills.deck_rules import validate_deck

ToolHandler = Callable[[dict[str, Any]], Any | Awaitable[Any]]

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

FORMAT_VALUES = [item.value for item in Format]
RAG_DOCUMENT_TOOL_MAX_LIMIT = 1_000
STRATEGY_SEARCH_DEFAULT_LIMIT = 40
STRATEGY_SEARCH_MAX_LIMIT = 100
META_DECK_SEARCH_DEFAULT_LIMIT = 40
RULES_SEARCH_DEFAULT_LIMIT = 20


@dataclass(frozen=True)
class McpToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def signature(self) -> str:
        properties = self.input_schema.get("properties", {})
        return f"{self.name}({', '.join(properties)})"


class DeckBuilderMcpServer:
    """Request-scoped MCP tool server for deck-building operations.

    The server owns all tool execution that can touch RAG, corpus-backed card data, or
    deterministic validation. Agent-facing wrappers should call tools through
    this registry instead of importing those dependencies directly.
    """

    def __init__(
        self,
        retriever: RagRetriever | None = None,
    ) -> None:
        self.retriever = retriever
        self._tools = _tool_definitions()
        self._handlers: dict[str, ToolHandler] = {
            "search_rag_text": self._tool_search_rag_text,
            "search_rag_vector": self._tool_search_rag_vector,
            "search_rag_metadata_prefixes": self._tool_search_rag_metadata_prefixes,
            "search_strategy": self._tool_search_strategy,
            "search_meta_decks": self._tool_search_meta_decks,
            "search_rules": self._tool_search_rules,
            "search_card_corpus": self._tool_search_card_corpus,
            "search_cards_scryfall": self._tool_search_cards_corpus,
            "lookup_card": self._tool_lookup_card_corpus,
            "validate_deck_cards": self._tool_validate_deck_cards,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        return [tool.as_dict() for tool in self._tools.values()]

    def tool_signatures(self, names: tuple[str, ...] | list[str]) -> list[str]:
        return [self._tools[name].signature() for name in names if name in self._tools]

    def call_tool_sync(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        result = self._call_handler(name, arguments or {})
        if isawaitable(result):
            raise TypeError(f"MCP tool {name!r} is async; use call_tool().")
        return result

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        result = self._call_handler(name, arguments or {})
        if isawaitable(result):
            return await result
        return result

    def _call_handler(self, name: str, arguments: dict[str, Any]) -> Any | Awaitable[Any]:
        try:
            handler = self._handlers[name]
        except KeyError as exc:
            raise ValueError(f"Unknown MCP tool: {name}") from exc
        return handler(arguments)

    def _require_retriever(self) -> RagRetriever:
        if self.retriever is None:
            raise RuntimeError("MCP RAG tools require a RagRetriever.")
        return self.retriever

    def _tool_search_rag_text(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        retriever = self._require_retriever()
        documents = retriever.search_text(
            query=_string_arg(arguments, "query"),
            limit=_int_arg(arguments, "limit", 5),
            source=_optional_string_arg(arguments.get("source")),
            metadata_filters=_metadata_filters_arg(arguments.get("metadata_filters")),
        )
        return [document_payload(document) for document in documents]

    def _tool_search_rag_vector(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        retriever = self._require_retriever()
        documents = retriever.search(
            query=_string_arg(arguments, "query"),
            limit=_int_arg(arguments, "limit", 5),
            source=_optional_string_arg(arguments.get("source")),
        )
        return [document_payload(document) for document in documents]

    def _tool_search_rag_metadata_prefixes(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        retriever = self._require_retriever()
        documents = retriever.search_by_metadata_prefix(
            source=_string_arg(arguments, "source"),
            metadata_key=_string_arg(arguments, "metadata_key"),
            prefixes=_string_list_arg(arguments.get("prefixes")),
            limit=_int_arg(arguments, "limit", 10),
        )
        return [document_payload(document) for document in documents]

    def _tool_search_strategy(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        query = _string_arg(arguments, "query")
        mtg_format = _format_arg(arguments.get("mtg_format") or arguments.get("format"))
        limit = min(_int_arg(arguments, "limit", STRATEGY_SEARCH_DEFAULT_LIMIT), STRATEGY_SEARCH_MAX_LIMIT)
        article_limit = max(1, (limit * 3) // 4)
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
        return [document_payload(document) for document in _dedupe_documents(documents)[:limit]]

    def _tool_search_meta_decks(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            document_payload(document)
            for document in self._search_strategy_source(
                query=_string_arg(arguments, "query"),
                source="mtgdecks_meta_decks",
                mtg_format=_format_arg(arguments.get("mtg_format") or arguments.get("format")),
                limit=min(
                    _int_arg(arguments, "limit", META_DECK_SEARCH_DEFAULT_LIMIT),
                    STRATEGY_SEARCH_MAX_LIMIT,
                ),
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
        return self._require_retriever().search_text(
            query=query,
            limit=limit,
            source=source,
            metadata_filters=metadata_filters,
        )

    def _tool_search_rules(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            document_payload(document)
            for document in self._require_retriever().search_text(
                query=_string_arg(arguments, "intent"),
                limit=min(_int_arg(arguments, "limit", RULES_SEARCH_DEFAULT_LIMIT), STRATEGY_SEARCH_MAX_LIMIT),
                source="mtg_comprehensive_rules",
            )
        ]

    def _tool_search_card_corpus(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        query = _string_arg(arguments, "query")
        limit = _int_arg(arguments, "limit", 20)
        request = _request_arg(arguments.get("request"))
        mtg_format = _format_arg(arguments.get("mtg_format") or arguments.get("format"))
        search_limit = max(limit * 6, 80)
        retriever = self._require_retriever()
        documents = _dedupe_documents(
            [
                *retriever.search(query=query, limit=search_limit, source="scryfall_bulk"),
                *retriever.search_text(query=query, limit=search_limit, source="scryfall_bulk"),
            ]
        )
        ranked: list[tuple[float, RetrievedDocument]] = []
        for document in documents:
            if not _is_allowed_card_document(document, query=query, mtg_format=mtg_format, request=request):
                continue
            ranked.append((_card_relevance_score(document, query=query, request=request), document))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [document_payload(document) for _, document in ranked[:limit]]

    def _tool_search_cards_corpus(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        query = _string_arg(arguments, "query")
        limit = _int_arg(arguments, "limit", 20)
        request = _request_arg(arguments.get("request"))
        mtg_format = _format_arg(arguments.get("mtg_format") or arguments.get("format"))
        return self._tool_search_card_corpus(
            {
                "query": query,
                "limit": limit,
                "request": request.model_dump(mode="json") if request is not None else None,
                "mtg_format": mtg_format.value if mtg_format is not None else None,
            }
        )

    def _tool_lookup_card_corpus(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = _string_arg(arguments, "name")
        results = self._tool_search_card_corpus(
            {
                "query": query,
                "limit": 1,
                "mtg_format": _optional_string_arg(arguments.get("mtg_format")),
                "request": arguments.get("request"),
            }
        )
        if results:
            return results[0]
        raise ValueError(f"No card corpus entry found for: {query}")

    def _tool_validate_deck_cards(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return validate_deck(
            cards=_cards_arg(arguments.get("cards")),
            mtg_format=_format_arg(arguments.get("mtg_format") or arguments.get("format")) or Format.casual,
        ).model_dump()


def document_payload(document: RetrievedDocument) -> dict[str, Any]:
    return {
        "title": document.title,
        "content": document.content,
        "source": document.source,
        "metadata": document.metadata,
        "score": document.score,
    }


def _document_price_usd(document: RetrievedDocument) -> float | None:
    price = document.metadata.get("price_usd")
    if isinstance(price, (int, float)):
        return float(price)
    price = document.metadata.get("estimated_price_usd")
    if isinstance(price, (int, float)):
        return float(price)
    return None


def payload_to_retrieved_document(payload: Any) -> RetrievedDocument | None:
    if not isinstance(payload, dict):
        return None
    title = payload.get("title")
    content = payload.get("content")
    source = payload.get("source")
    metadata = payload.get("metadata")
    if not isinstance(title, str) or not isinstance(content, str) or not isinstance(source, str):
        return None
    if not isinstance(metadata, dict):
        metadata = {}
    score = payload.get("score")
    return RetrievedDocument(
        title=title,
        content=content,
        source=source,
        metadata=metadata,
        score=float(score) if isinstance(score, int | float) else None,
    )


def _tool_definitions() -> dict[str, McpToolDefinition]:
    return {
        "search_rag_text": McpToolDefinition(
            name="search_rag_text",
            description="Search indexed RAG documents by text terms.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": RAG_DOCUMENT_TOOL_MAX_LIMIT},
                    "source": {"type": "string"},
                    "metadata_filters": {
                        "type": "object",
                        "additionalProperties": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        ),
        "search_rag_vector": McpToolDefinition(
            name="search_rag_vector",
            description="Search indexed RAG documents by semantic vector similarity.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": RAG_DOCUMENT_TOOL_MAX_LIMIT},
                    "source": {"type": "string"},
                },
            },
        ),
        "search_rag_metadata_prefixes": McpToolDefinition(
            name="search_rag_metadata_prefixes",
            description="Search indexed RAG documents by metadata prefix.",
            input_schema={
                "type": "object",
                "required": ["source", "metadata_key", "prefixes"],
                "properties": {
                    "source": {"type": "string"},
                    "metadata_key": {"type": "string"},
                    "prefixes": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": RAG_DOCUMENT_TOOL_MAX_LIMIT},
                },
            },
        ),
        "search_strategy": McpToolDefinition(
            name="search_strategy",
            description="Search strategy articles and meta deck snapshots for a deck request.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "mtg_format": {"type": "string", "enum": FORMAT_VALUES},
                    "limit": {"type": "integer", "minimum": 1, "maximum": STRATEGY_SEARCH_MAX_LIMIT},
                },
            },
        ),
        "search_meta_decks": McpToolDefinition(
            name="search_meta_decks",
            description="Search MTGDecks archetype and top-deck snapshots.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "mtg_format": {"type": "string", "enum": FORMAT_VALUES},
                    "limit": {"type": "integer", "minimum": 1, "maximum": STRATEGY_SEARCH_MAX_LIMIT},
                },
            },
        ),
        "search_rules": McpToolDefinition(
            name="search_rules",
            description="Search Magic comprehensive rules RAG documents.",
            input_schema={
                "type": "object",
                "required": ["intent"],
                "properties": {
                    "intent": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": STRATEGY_SEARCH_MAX_LIMIT},
                },
            },
        ),
        "search_card_corpus": McpToolDefinition(
            name="search_card_corpus",
            description="Search the indexed Scryfall bulk card corpus with deck-request filters.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "mtg_format": {"type": "string", "enum": FORMAT_VALUES},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "request": {"type": "object"},
                },
            },
        ),
        "search_cards_scryfall": McpToolDefinition(
            name="search_cards_scryfall",
            description="Search the indexed Scryfall card corpus for current card facts, prices, and legality.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
        ),
        "lookup_card": McpToolDefinition(
            name="lookup_card",
            description="Look up one exact card name in the indexed Scryfall card corpus.",
            input_schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
        ),
        "validate_deck_cards": McpToolDefinition(
            name="validate_deck_cards",
            description="Validate deck size and copy limits for a format.",
            input_schema={
                "type": "object",
                "required": ["cards", "mtg_format"],
                "properties": {
                    "cards": {"type": "array", "items": {"type": "object"}},
                    "mtg_format": {"type": "string", "enum": FORMAT_VALUES},
                },
            },
        ),
    }


def _string_arg(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MCP tool argument {key!r} must be a non-empty string.")
    return value.strip()


def _optional_string_arg(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Optional string MCP argument must be a string.")
    stripped = value.strip()
    return stripped or None


def _int_arg(arguments: dict[str, Any], key: str, default: int) -> int:
    value = arguments.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MCP tool argument {key!r} must be an integer.") from exc
    return max(1, parsed)


def _string_list_arg(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _metadata_filters_arg(value: Any) -> dict[str, list[str]] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("metadata_filters must be an object.")
    filters: dict[str, list[str]] = {}
    for key, raw_values in value.items():
        if not isinstance(key, str):
            continue
        filters[key] = _string_list_arg(raw_values)
    return filters or None


def _format_arg(value: Any) -> Format | None:
    if value is None or value == "":
        return None
    if isinstance(value, Format):
        return value
    try:
        return Format(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported MTG format for MCP tool call: {value!r}") from exc


def _request_arg(value: Any) -> DeckRequest | None:
    if value is None:
        return None
    if isinstance(value, DeckRequest):
        return value
    if isinstance(value, dict):
        return DeckRequest.model_validate(value)
    raise ValueError("request must be a DeckRequest payload.")


def _cards_arg(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


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
            price = _document_price_usd(document)
            if price is not None:
                score += max(0.0, 3.0 - min(price, 30.0) / 10.0)

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
