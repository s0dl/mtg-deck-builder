from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.rag.chunking import chunk_text, join_card_fields
from app.rag.repository import RagDocumentInput


def build_card_documents(card: dict[str, Any]) -> list[RagDocumentInput]:
    oracle_id = card.get("oracle_id") or card.get("id")
    name = card.get("name", "Unknown Card")
    content = join_card_fields(card)
    chunks = chunk_text(content)
    documents: list[RagDocumentInput] = []

    for index, chunk in enumerate(chunks):
        documents.append(
            RagDocumentInput(
                source="scryfall_bulk",
                source_id=f"{oracle_id}:{index}",
                title=name,
                content=chunk,
                metadata={
                    "card_id": card.get("id"),
                    "oracle_id": oracle_id,
                    "name": name,
                    "colors": card.get("colors", []),
                    "color_identity": card.get("color_identity", []),
                    "mana_value": card.get("cmc"),
                    "type_line": card.get("type_line"),
                    "keywords": card.get("keywords", []),
                    "legalities": card.get("legalities", {}),
                },
            )
        )

    return documents


def load_scryfall_bulk_file(path: Path) -> list[RagDocumentInput]:
    with path.open("r", encoding="utf-8") as handle:
        cards = json.load(handle)

    documents: list[RagDocumentInput] = []
    for card in cards:
        if card.get("layout") in {"art_series", "token", "double_faced_token"}:
            continue
        documents.extend(build_card_documents(card))

    return documents


def build_strategy_document(
    source: str,
    source_id: str,
    title: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> list[RagDocumentInput]:
    return [
        RagDocumentInput(
            source=source,
            source_id=f"{source_id}:{index}",
            title=title,
            content=chunk,
            metadata=metadata or {},
        )
        for index, chunk in enumerate(chunk_text(content))
    ]
