import logging
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import log_extra
from app.rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)


def price_usd(card: dict[str, Any]) -> float | None:
    prices = card.get("prices") or {}
    price = prices.get("usd") or prices.get("usd_foil")
    if price is None:
        return None
    try:
        return float(price)
    except (TypeError, ValueError):
        return None


def scryfall_card_to_document(card: dict[str, Any]) -> RetrievedDocument:
    name = str(card.get("name") or "")
    faces = card.get("card_faces") or []
    oracle_text = card.get("oracle_text") or "\n".join(
        str(face.get("oracle_text") or "") for face in faces
    )
    type_line = card.get("type_line") or " // ".join(
        str(face.get("type_line") or "") for face in faces
    )
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
            "scryfall_uri": card.get("scryfall_uri"),
        },
    )


class ScryfallClient:
    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.scryfall_api_base_url).rstrip("/")

    async def get_card_named(self, name: str) -> dict[str, Any]:
        logger.info("Scryfall card lookup started", extra=log_extra(name=name))
        async with httpx.AsyncClient(base_url=self.base_url, timeout=15) as client:
            response = await client.get("/cards/named", params={"exact": name})
            response.raise_for_status()
            payload = response.json()
            logger.info("Scryfall card lookup completed", extra=log_extra(name=name))
            return payload

    async def search_cards(
        self,
        query: str,
        page: int = 1,
        order: str = "edhrec",
        unique: str = "cards",
        include_extras: bool = False,
    ) -> dict[str, Any]:
        logger.info("Scryfall search started", extra=log_extra(query=query, page=page))
        async with httpx.AsyncClient(base_url=self.base_url, timeout=20) as client:
            response = await client.get(
                "/cards/search",
                params={
                    "q": query,
                    "page": page,
                    "order": order,
                    "unique": unique,
                    "include_extras": str(include_extras).lower(),
                },
            )
            response.raise_for_status()
            payload = response.json()
            logger.info(
                "Scryfall search completed",
                extra=log_extra(query=query, page=page, result_count=len(payload.get("data", []))),
            )
            return payload
