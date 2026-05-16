from __future__ import annotations

from typing import Any

from app.rag.retriever import RetrievedDocument


def document_to_model_context(document: RetrievedDocument) -> dict[str, Any]:
    metadata = document.metadata
    return {
        "title": document.title,
        "source": document.source,
        "content": document.content[:900],
        "metadata": {
            "name": metadata.get("name"),
            "colors": metadata.get("colors"),
            "color_identity": metadata.get("color_identity"),
            "mana_value": metadata.get("mana_value"),
            "type_line": metadata.get("type_line"),
            "legalities": metadata.get("legalities"),
            "category": metadata.get("category"),
            "url": metadata.get("url"),
            "rule_number": metadata.get("rule_number"),
        },
    }
