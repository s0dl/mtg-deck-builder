from app.agent.tools import DeckAgentTools
from app.mcp.server import DeckBuilderMcpServer
from app.models.deck import DeckRequest, Format
from app.rag.retriever import RetrievedDocument


class FakeRetriever:
    def __init__(
        self,
        documents: list[RetrievedDocument],
        text_documents: list[RetrievedDocument] | None = None,
    ) -> None:
        self.documents = documents
        self.text_documents = text_documents if text_documents is not None else []
        self.last_limit: int | None = None
        self.text_limits_by_source: dict[str | None, list[int]] = {}

    def search(self, query: str, limit: int = 5, source: str | None = None) -> list[RetrievedDocument]:
        self.last_limit = limit
        return [document for document in self.documents if source is None or document.source == source][:limit]

    def search_text(
        self,
        query: str,
        limit: int = 5,
        source: str | None = None,
        metadata_filters: dict[str, list[str]] | None = None,
    ):
        self.text_limits_by_source.setdefault(source, []).append(limit)
        documents = [document for document in self.text_documents if source is None or document.source == source]
        if metadata_filters:
            for key, values in metadata_filters.items():
                normalized_values = {value.lower() for value in values}
                documents = [
                    document
                    for document in documents
                    if str(document.metadata.get(key) or "").lower() in normalized_values
                ]
        return documents[:limit]

    def search_by_metadata_prefix(
        self,
        source: str,
        metadata_key: str,
        prefixes: list[str],
        limit: int = 10,
    ) -> list[RetrievedDocument]:
        return [
            document
            for document in self.text_documents
            if document.source == source
            and any(str(document.metadata.get(metadata_key) or "").startswith(prefix) for prefix in prefixes)
        ][:limit]


def card_document(
    name: str,
    *,
    type_line: str = "Instant",
    colors: list[str] | None = None,
    color_identity: list[str] | None = None,
    legalities: dict[str, str] | None = None,
    content: str = "",
    score: float = 0.0,
) -> RetrievedDocument:
    return RetrievedDocument(
        title=name,
        content=content or f"{name} card text",
        source="scryfall_bulk",
        metadata={
            "name": name,
            "type_line": type_line,
            "colors": colors or [],
            "color_identity": color_identity if color_identity is not None else colors or [],
            "legalities": legalities or {"modern": "legal"},
            "estimated_price_usd": 1.0,
        },
        score=score,
    )


def test_search_card_corpus_filters_legality_colors_and_lands() -> None:
    retriever = FakeRetriever(
        [
            card_document("Monastery Swiftspear", colors=["R"], content="prowess haste damage", score=0.2),
            card_document("Drown in the Loch", colors=["U", "B"], color_identity=["U", "B"], score=0.9),
            card_document("Ponder", colors=["U"], legalities={"modern": "not_legal"}, score=0.9),
            card_document("Steam Vents", type_line="Land - Island Mountain", color_identity=["U", "R"], score=0.9),
        ]
    )
    tools = DeckAgentTools(retriever=retriever)

    results = tools.search_card_corpus(
        query="modern izzet prowess cheap threat damage",
        mtg_format=Format.modern,
        request=DeckRequest(format=Format.modern, colors=["U", "R"], strategy="prowess damage"),
        limit=10,
    )

    assert [result["title"] for result in results] == ["Monastery Swiftspear"]
    assert retriever.last_limit == 80


def test_search_card_corpus_land_queries_return_budget_lands_first() -> None:
    cheap_land = card_document(
        "Shivan Reef",
        type_line="Land",
        color_identity=["U", "R"],
        content="blue red pain land mana fixing",
        score=0.1,
    )
    cheap_land.metadata["estimated_price_usd"] = 0.75
    expensive_land = card_document(
        "Steam Vents",
        type_line="Land - Island Mountain",
        color_identity=["U", "R"],
        content="blue red shock land mana fixing",
        score=0.1,
    )
    expensive_land.metadata["estimated_price_usd"] = 20.0
    spell = card_document("Lightning Bolt", colors=["R"], content="damage", score=1.0)
    retriever = FakeRetriever([expensive_land, spell, cheap_land])
    tools = DeckAgentTools(retriever=retriever)

    results = tools.search_card_corpus(
        query="modern land blue red mana fixing cheap budget",
        mtg_format=Format.modern,
        request=DeckRequest(format=Format.modern, colors=["U", "R"], budget_usd=100),
        limit=10,
    )

    assert [result["title"] for result in results] == ["Shivan Reef", "Steam Vents"]


def test_search_strategy_includes_meta_deck_documents() -> None:
    article = RetrievedDocument(
        title="Modern Izzet Prowess Guide",
        content="Tempo strategy",
        source="mtgdecks_articles",
        metadata={"format": "modern"},
    )
    meta_deck = RetrievedDocument(
        title="Izzet Prowess (Modern)",
        content="Top deck cards: Monastery Swiftspear",
        source="mtgdecks_meta_decks",
        metadata={"format": "modern", "archetype": "Izzet Prowess"},
    )
    retriever = FakeRetriever([], text_documents=[article, meta_deck])
    tools = DeckAgentTools(retriever=retriever)

    results = tools.search_strategy(query="modern izzet prowess", mtg_format=Format.modern, limit=4)

    assert [result["source"] for result in results] == ["mtgdecks_articles", "mtgdecks_meta_decks"]


def test_search_strategy_requests_larger_article_context_by_default() -> None:
    retriever = FakeRetriever([], text_documents=[])
    tools = DeckAgentTools(retriever=retriever)

    tools.search_strategy(query="modern izzet prowess", mtg_format=Format.modern)

    assert retriever.text_limits_by_source["mtgdecks_articles"] == [30]
    assert retriever.text_limits_by_source["mtgdecks_meta_decks"] == [10]


def test_search_card_corpus_combines_vector_and_text_results() -> None:
    vector_card = card_document("Monastery Swiftspear", colors=["R"], content="prowess haste")
    text_card = card_document("Slickshot Show-Off", colors=["R"], content="flying haste prowess")
    retriever = FakeRetriever([vector_card], text_documents=[text_card])
    tools = DeckAgentTools(retriever=retriever)

    results = tools.search_card_corpus(
        query="modern red prowess",
        mtg_format=Format.modern,
        request=DeckRequest(format=Format.modern, colors=["R"], strategy="prowess"),
        limit=10,
    )

    assert {result["title"] for result in results} == {"Monastery Swiftspear", "Slickshot Show-Off"}


def test_evaluate_deck_candidates_scores_request_fit() -> None:
    tools = DeckAgentTools(retriever=FakeRetriever([]))

    results = tools.evaluate_deck_candidates(
        [
            {
                "name": "Monastery Swiftspear",
                "type_line": "Creature - Human Monk",
                "mana_value": 1,
                "colors": ["R"],
                "color_identity": ["R"],
                "legalities": {"modern": "legal"},
                "content": "prowess haste",
            },
            {
                "name": "Ponder",
                "type_line": "Sorcery",
                "colors": ["U"],
                "color_identity": ["U"],
                "legalities": {"modern": "not_legal"},
            },
        ],
        DeckRequest(format=Format.modern, colors=["R"], playstyle="aggro", strategy="prowess"),
    )

    assert results[0]["name"] == "Monastery Swiftspear"
    assert results[0]["is_playable"] is True
    assert results[-1]["name"] == "Ponder"
    assert results[-1]["is_playable"] is False


def test_curate_context_notes_dedupes_and_limits_documents() -> None:
    tools = DeckAgentTools(retriever=FakeRetriever([]))
    documents = [
        {"title": "Guide", "source": "mtgdecks_articles", "content": " play cheap threats and burn "},
        {"title": "Guide", "source": "mtgdecks_articles", "content": "play cheap threats and burn"},
        {"title": "Rules", "source": "mtg_comprehensive_rules", "content": "Deck size minimums apply."},
    ]

    notes = tools.curate_context_notes(documents, max_notes=1)

    assert notes == ["Guide (mtgdecks_articles): play cheap threats and burn"]


def test_mcp_server_lists_all_agent_tool_categories() -> None:
    server = DeckBuilderMcpServer(retriever=FakeRetriever([]))

    tool_names = {tool["name"] for tool in server.list_tools()}

    assert {
        "search_rag_text",
        "search_rag_vector",
        "search_rag_metadata_prefixes",
        "search_strategy",
        "search_meta_decks",
        "search_rules",
        "search_card_corpus",
        "search_cards_scryfall",
        "lookup_card",
        "validate_deck_cards",
    }.issubset(tool_names)
