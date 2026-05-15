from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.logging import log_extra
from app.core.openai_agents import run_structured_openai_agent
from app.models.deck import DeckRequest
from app.rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class DeckCardOutput(BaseModel):
    name: str
    count: int = Field(ge=1)
    role: str


class DeckBuilderOutput(BaseModel):
    title: str
    explanation: str
    cards: list[DeckCardOutput] = Field(min_length=1)


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
        instructions = (
            "You are a Magic: The Gathering deck-building assistant. Build flexible, "
            "coherent decks from the provided candidate cards and strategy/rules context. "
            "Use the rules context to obey format deck size and copy limits. Prefer card "
            "synergy over rigid templates. Choose a land count from the land guidance based "
            "on curve, colors, and strategy; do not always use the default. Use only "
            "candidate card names plus basic lands."
        )
        input_payload = {
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
        result = await run_structured_openai_agent(
            settings=self.settings,
            name="MTG deck builder",
            instructions=instructions,
            input_payload=input_payload,
            output_type=DeckBuilderOutput,
        )
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
