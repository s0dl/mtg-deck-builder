from app.agent.tools import DeckAgentTools
from app.models.deck import DeckRequest, Format
from app.rag.retriever import RetrievedDocument


class FakeRetriever:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self.documents = documents
        self.last_limit: int | None = None

    def search(self, query: str, limit: int = 5, source: str | None = None) -> list[RetrievedDocument]:
        self.last_limit = limit
        return self.documents[:limit]

    def search_text(self, *args, **kwargs):
        return []


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
    tools = DeckAgentTools(retriever=retriever, scryfall=None)  # type: ignore[arg-type]

    results = tools.search_card_corpus(
        query="modern izzet prowess cheap threat damage",
        mtg_format=Format.modern,
        request=DeckRequest(format=Format.modern, colors=["U", "R"], strategy="prowess damage"),
        limit=10,
    )

    assert [result["title"] for result in results] == ["Monastery Swiftspear"]
    assert retriever.last_limit == 50


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
    tools = DeckAgentTools(retriever=retriever, scryfall=None)  # type: ignore[arg-type]

    results = tools.search_card_corpus(
        query="modern land blue red mana fixing cheap budget",
        mtg_format=Format.modern,
        request=DeckRequest(format=Format.modern, colors=["U", "R"], budget_usd=100),
        limit=10,
    )

    assert [result["title"] for result in results] == ["Shivan Reef", "Steam Vents"]
