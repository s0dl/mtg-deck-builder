from app.rag.retriever import RetrievedDocument, _score_text_match
from app.api.decks import _dedupe_documents, _rules_prefixes_for_request
from app.models.deck import DeckRequest, Format


def test_rules_prefixes_include_commander_rules_for_commander() -> None:
    prefixes = _rules_prefixes_for_request(DeckRequest(format=Format.commander))

    assert "903.5a" in prefixes
    assert "100.2a" in prefixes


def test_dedupe_documents_removes_duplicate_rule_context() -> None:
    first = RetrievedDocument(
        title="Rule 100.2",
        content="To play, each player needs their own deck.",
        source="mtg_comprehensive_rules",
        metadata={"rule_number": "100.2"},
    )
    second = RetrievedDocument(
        title="Rule 100.2",
        content="To play, each player needs their own deck.",
        source="mtg_comprehensive_rules",
        metadata={"rule_number": "100.2"},
    )

    assert _dedupe_documents([first, second]) == [first]


def test_score_text_match_prioritizes_title_and_metadata() -> None:
    boros_score = _score_text_match(
        title="Boros Mobilize in MTG Standard",
        content="Aggro deck guide.",
        terms=["modern", "izzet", "blue", "red"],
        metadata={"format": "standard"},
    )
    izzet_score = _score_text_match(
        title="Modern Izzet Prowess Deck Guide",
        content="Blue red tempo strategy.",
        terms=["modern", "izzet", "blue", "red"],
        metadata={"format": "modern"},
    )

    assert izzet_score > boros_score
