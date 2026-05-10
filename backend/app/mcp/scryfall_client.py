from typing import Any

import httpx

from app.core.config import get_settings


class ScryfallClient:
    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.scryfall_api_base_url).rstrip("/")

    async def get_card_named(self, name: str) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=15) as client:
            response = await client.get("/cards/named", params={"exact": name})
            response.raise_for_status()
            return response.json()

    async def search_cards(self, query: str, page: int = 1) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=20) as client:
            response = await client.get("/cards/search", params={"q": query, "page": page})
            response.raise_for_status()
            return response.json()
