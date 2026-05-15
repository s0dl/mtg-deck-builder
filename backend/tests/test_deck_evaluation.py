from app.models.deck import DeckRequest, Format
from app.rag.retriever import RetrievedDocument
from app.skills.deck_evaluation import evaluate_candidate_document, rank_candidate_documents


def card_document(
    name: str,
    *,
    type_line: str = "Instant",
    mana_value: int = 1,
    colors: list[str] | None = None,
    color_identity: list[str] | None = None,
    legalities: dict[str, str] | None = None,
    content: str = "",
    price: float | None = 1.0,
) -> RetrievedDocument:
    return RetrievedDocument(
        title=name,
        content=content or f"{name} card text",
        source="scryfall_bulk",
        metadata={
            "name": name,
            "type_line": type_line,
            "mana_value": mana_value,
            "colors": colors or [],
            "color_identity": color_identity if color_identity is not None else colors or [],
            "legalities": legalities if legalities is not None else {"modern": "legal", "commander": "legal"},
            "estimated_price_usd": price,
        },
        score=0.0,
    )


def test_rank_candidate_documents_prefers_curve_and_strategy_fit() -> None:
    request = DeckRequest(
        format=Format.modern,
        colors=["R"],
        playstyle="aggro",
        strategy="prowess damage",
    )
    expensive_top_end = card_document(
        "Six Mana Dragon",
        type_line="Creature - Dragon",
        mana_value=6,
        colors=["R"],
        content="Flying",
    )
    cheap_synergy_card = card_document(
        "Monastery Swiftspear",
        type_line="Creature - Human Monk",
        mana_value=1,
        colors=["R"],
        content="Haste prowess damage",
    )

    ranked = rank_candidate_documents([expensive_top_end, cheap_synergy_card], request)

    assert [document.title for document in ranked] == ["Monastery Swiftspear", "Six Mana Dragon"]


def test_rank_candidate_documents_filters_legality_color_identity_and_avoid_terms() -> None:
    request = DeckRequest(
        format=Format.modern,
        colors=["U", "R"],
        avoid=["discard"],
    )
    documents = [
        card_document("Lightning Bolt", colors=["R"], content="damage"),
        card_document("Drown in the Loch", colors=["U", "B"], color_identity=["U", "B"]),
        card_document("Ponder", colors=["U"], legalities={"modern": "not_legal"}),
        card_document("Mind Rot", colors=["B"], color_identity=["B"], content="Target player discards cards."),
    ]

    ranked = rank_candidate_documents(documents, request)

    assert [document.title for document in ranked] == ["Lightning Bolt"]


def test_evaluate_candidate_document_penalizes_expensive_cards_for_budget() -> None:
    request = DeckRequest(format=Format.modern, colors=["U", "R"], budget_usd=40)
    cheap_land = card_document("Shivan Reef", type_line="Land", color_identity=["U", "R"], price=0.75)
    expensive_land = card_document("Steam Vents", type_line="Land - Island Mountain", color_identity=["U", "R"], price=20.0)

    cheap_evaluation = evaluate_candidate_document(cheap_land, request)
    expensive_evaluation = evaluate_candidate_document(expensive_land, request)

    assert cheap_evaluation.score > expensive_evaluation.score
