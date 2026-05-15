from pathlib import Path

from app.rag.ingestion import load_comprehensive_rules_file


def test_load_comprehensive_rules_file_chunks_numbered_rules(tmp_path: Path) -> None:
    rules_file = tmp_path / "rules.txt"
    rules_file.write_text(
        "\n".join(
            [
                "Magic: The Gathering Comprehensive Rules",
                "100. General",
                "100.1. These Magic rules apply to any Magic game.",
                "100.2. To play, each player needs their own deck.",
                "101. The Magic Golden Rules",
                "101.1. Whenever a card's text directly contradicts these rules, the card takes precedence.",
            ]
        ),
        encoding="utf-8",
    )

    documents = load_comprehensive_rules_file(rules_file)

    assert [document.source_id for document in documents] == [
        "100:0",
        "100.1:0",
        "100.2:0",
        "101:0",
        "101.1:0",
    ]
    assert documents[0].source == "mtg_comprehensive_rules"
    assert documents[0].metadata["rule_number"] == "100"
    assert "minimum" not in documents[0].metadata
