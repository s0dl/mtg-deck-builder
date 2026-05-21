from __future__ import annotations

from typing import Any

from app.rag.retriever import RetrievedDocument


def price_usd(card: dict[str, Any]) -> float | None:
    prices = card.get("prices") or {}
    price = prices.get("usd") or prices.get("usd_foil") or prices.get("usd_etched")
    if price is None:
        return None
    try:
        return float(price)
    except (TypeError, ValueError):
        return None


def scryfall_card_to_document(card: dict[str, Any]) -> RetrievedDocument:
    name = str(card.get("name") or "")
    faces = card.get("card_faces") or []
    oracle_text = card.get("oracle_text") or "\n".join(str(face.get("oracle_text") or "") for face in faces)
    type_line = card.get("type_line") or " // ".join(str(face.get("type_line") or "") for face in faces)
    colors = card.get("colors")
    if colors is None and faces:
        colors = sorted({color for face in faces for color in face.get("colors") or []})

    return RetrievedDocument(
        title=name,
        content="\n".join(part for part in [type_line, oracle_text] if part),
        source="scryfall_live",
        metadata={
            "name": name,
            "mana_value": card.get("cmc"),
            "type_line": type_line,
            "colors": colors or [],
            "color_identity": card.get("color_identity") or [],
            "legalities": card.get("legalities") or {},
            "estimated_price_usd": price_usd(card),
            "price_usd": price_usd(card),
            "scryfall_uri": card.get("scryfall_uri"),
        },
    )
