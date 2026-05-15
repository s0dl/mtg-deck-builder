from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import log_extra
from app.models.deck import DeckRequest
from app.rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

DECK_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "explanation", "cards"],
    "properties": {
        "title": {"type": "string"},
        "explanation": {"type": "string"},
        "cards": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "count", "role"],
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1},
                    "role": {"type": "string"},
                },
            },
        },
    },
}


class OpenAIDeckBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate(
        self,
        request: DeckRequest,
        card_context: list[RetrievedDocument],
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
        land_guidance: dict[str, int],
    ) -> dict[str, Any]:
        payload = {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are a Magic: The Gathering deck-building assistant. Build flexible, "
                        "coherent decks from the provided candidate cards and strategy/rules context. "
                        "Use the rules context to obey format deck size and copy limits. Prefer card "
                        "synergy over rigid templates. Choose a land count from the land guidance based "
                        "on curve, colors, and strategy; do not always use the default. Use only "
                        "candidate card names plus basic lands. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": request.model_dump(mode="json"),
                            "land_guidance": land_guidance,
                            "candidate_cards": [
                                _document_to_model_context(document) for document in card_context[:180]
                            ],
                            "rules_context": [
                                _document_to_model_context(document) for document in rules_context[:12]
                            ],
                            "strategy_context": [
                                _document_to_model_context(document) for document in strategy_context[:16]
                            ],
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "mtg_deck",
                    "strict": True,
                    "schema": DECK_RESPONSE_SCHEMA,
                }
            },
        }

        logger.info(
            "OpenAI deck generation request started",
            extra=log_extra(
                model=self.settings.openai_model,
                card_context_count=len(card_context),
                rules_context_count=len(rules_context),
                strategy_context_count=len(strategy_context),
            ),
        )
        async with httpx.AsyncClient(
            base_url=self.settings.openai_base_url,
            timeout=45,
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            response = await client.post("/responses", json=payload)
            response.raise_for_status()

        result = _extract_json_response(response.json())
        logger.info(
            "OpenAI deck generation request completed",
            extra=log_extra(model=self.settings.openai_model, card_count=len(result.get("cards", []))),
        )
        return result


def _document_to_model_context(document: RetrievedDocument) -> dict[str, Any]:
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


def _extract_json_response(response_json: dict[str, Any]) -> dict[str, Any]:
    if isinstance(response_json.get("output_text"), str):
        return json.loads(response_json["output_text"])

    for output_item in response_json.get("output", []):
        for content_item in output_item.get("content", []):
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                return json.loads(text)

    raise ValueError("OpenAI response did not include JSON text output.")
