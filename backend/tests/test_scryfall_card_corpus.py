import json
from pathlib import Path

from app.rag.ingestion import load_scryfall_card_corpus_file
from scripts.download_scryfall_card_corpus import default_cards_download_uri


def test_default_cards_download_uri_reads_bulk_index() -> None:
    uri = default_cards_download_uri(
        {
            "data": [
                {"type": "oracle_cards", "download_uri": "https://example.test/oracle.json"},
                {"type": "default_cards", "download_uri": "https://example.test/default.json"},
            ]
        }
    )

    assert uri == "https://example.test/default.json"


def test_load_scryfall_card_corpus_file_builds_card_text_documents(tmp_path: Path) -> None:
    card_file = tmp_path / "cards.json"
    card_file.write_text(
        json.dumps(
            [
                {
                    "id": "card-1",
                    "oracle_id": "oracle-1",
                    "name": "Lightning Bolt",
                    "type_line": "Instant",
                    "oracle_text": "Lightning Bolt deals 3 damage to any target.",
                    "colors": ["R"],
                    "color_identity": ["R"],
                    "cmc": 1,
                    "keywords": [],
                    "legalities": {"modern": "legal"},
                    "layout": "normal",
                    "games": ["paper"],
                },
                {
                    "id": "token-1",
                    "name": "Goblin Token",
                    "layout": "token",
                    "games": ["paper"],
                },
            ]
        ),
        encoding="utf-8",
    )

    documents = load_scryfall_card_corpus_file(card_file)

    assert len(documents) == 1
    assert documents[0].source == "scryfall_bulk"
    assert documents[0].source_id == "oracle-1:0"
    assert documents[0].metadata["kind"] == "card_text"
    assert documents[0].metadata["name"] == "Lightning Bolt"
    assert "Mana value: 1" in documents[0].content
    assert "Color identity: R" in documents[0].content
    assert "Legal formats: modern" in documents[0].content
    assert "3 damage" in documents[0].content
