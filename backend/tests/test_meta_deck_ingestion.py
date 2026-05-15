import json
from pathlib import Path

from app.rag.ingestion import load_mtgdecks_meta_decks_file


def test_load_mtgdecks_meta_decks_file_builds_rag_documents(tmp_path: Path) -> None:
    meta_file = tmp_path / "meta.json"
    meta_file.write_text(
        json.dumps(
            [
                {
                    "format": "standard",
                    "name": "Izzet Prowess",
                    "url": "https://mtgdecks.net/Standard/izzet-prowess",
                    "metagame_share": 12.5,
                    "top_deck": {
                        "url": "https://mtgdecks.net/Standard/izzet-prowess-decklist-by-gnawe-2931400",
                        "cards": [
                            {"count": 4, "name": "Monastery Swiftspear"},
                            {"count": 4, "name": "Sleight of Hand"},
                        ],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    documents = load_mtgdecks_meta_decks_file(meta_file)

    assert len(documents) == 1
    assert documents[0].source == "mtgdecks_meta_decks"
    assert documents[0].source_id == "https://mtgdecks.net/Standard/izzet-prowess:0"
    assert documents[0].metadata["format"] == "standard"
    assert documents[0].metadata["archetype"] == "Izzet Prowess"
    assert documents[0].metadata["metagame_share"] == 12.5
    assert "Monastery Swiftspear" in documents[0].content
    assert "Sleight of Hand" in documents[0].metadata["card_names"]
