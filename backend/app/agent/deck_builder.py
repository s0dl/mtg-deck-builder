from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.agent.tools import DeckAgentTools
from app.core.config import Settings
from app.core.logging import log_extra
from app.llm.deck_builder import _document_to_model_context, _extract_json_response
from app.models.deck import DeckRequest
from app.rag.retriever import RetrievedDocument
from app.skills.deck_evaluation import rank_candidate_cards

logger = logging.getLogger(__name__)

AGENT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["thoughts", "strategy_queries", "rules_queries", "card_queries"],
    "properties": {
        "thoughts": {"type": "array", "items": {"type": "string"}},
        "strategy_queries": {"type": "array", "items": {"type": "string"}},
        "rules_queries": {"type": "array", "items": {"type": "string"}},
        "card_queries": {"type": "array", "items": {"type": "string"}},
    },
}

AGENT_SELECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "explanation", "selected_cards"],
    "properties": {
        "title": {"type": "string"},
        "explanation": {"type": "string"},
        "selected_cards": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "count", "role"],
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1, "maximum": 4},
                    "role": {"type": "string"},
                },
            },
        },
    },
}


class AbstractDeckAgent(ABC):
    """Shared orchestration contract for deck-building agents."""

    def __init__(self, settings: Settings, tools: DeckAgentTools) -> None:
        self.settings = settings
        self.tools = tools

    async def generate(
        self,
        request: DeckRequest,
        card_context: list[RetrievedDocument],
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
        land_guidance: dict[str, int],
    ) -> dict[str, Any]:
        steps: list[dict[str, str]] = [
            {
                "label": "GPT received initial RAG context",
                "detail": (
                    f"Loaded {len(strategy_context)} strategy documents, "
                    f"{len(rules_context)} rules documents, and {len(card_context)} candidate cards."
                ),
            }
        ]
        plan = await self._plan(request, card_context, rules_context, strategy_context)
        steps.append(
            {
                "label": "GPT tool plan",
                "detail": (
                    f"Planned {len(_string_list(plan.get('strategy_queries')))} strategy, "
                    f"{len(_string_list(plan.get('rules_queries')))} rules, and "
                    f"{len(_string_list(plan.get('card_queries')))} card corpus RAG searches."
                ),
            }
        )
        for thought in _string_list(plan.get("thoughts"))[:4]:
            steps.append({"label": self.thought_label, "detail": thought})

        tool_results = await self._run_tool_plan(request, plan, steps)
        result = await self._select_cards(
            request=request,
            card_context=card_context,
            rules_context=rules_context,
            strategy_context=strategy_context,
            tool_results=tool_results,
            land_guidance=land_guidance,
        )
        steps.append(
            {"label": "GPT card selection", "detail": f"Selected {len(result.get('selected_cards', []))} card names from RAG card context."}
        )
        result["agent_steps"] = steps
        result["tool_card_names"] = _tool_card_names(tool_results)
        result["tool_card_payloads"] = _tool_card_payloads(tool_results)
        return result

    @property
    @abstractmethod
    def thought_label(self) -> str:
        """Human-readable label for model reasoning snippets."""

    @abstractmethod
    async def _plan(
        self,
        request: DeckRequest,
        card_context: list[RetrievedDocument],
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
    ) -> dict[str, Any]:
        """Return a normalized agent plan matching AGENT_PLAN_SCHEMA."""

    @abstractmethod
    async def _run_tool_plan(
        self,
        request: DeckRequest,
        plan: dict[str, Any],
        steps: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Execute backend tools requested by the plan and return normalized tool results."""

    @abstractmethod
    async def _select_cards(
        self,
        request: DeckRequest,
        card_context: list[RetrievedDocument],
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
        tool_results: dict[str, Any],
        land_guidance: dict[str, int],
    ) -> dict[str, Any]:
        """Return selected cards matching AGENT_SELECTION_SCHEMA."""


class OllamaDeckAgent(AbstractDeckAgent):
    @property
    def thought_label(self) -> str:
        return "Agent thought"

    async def _plan(
        self,
        request: DeckRequest,
        card_context: list[RetrievedDocument],
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
    ) -> dict[str, Any]:
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "format": AGENT_PLAN_SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a constrained Magic: The Gathering deck-building agent. "
                        "Plan only the extra RAG tool calls needed before deck construction. "
                        "Use card corpus searches for candidate cards. Live Scryfall is reserved "
                        "for backend verification and price checks after validation."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": request.model_dump(mode="json"),
                            "available_tools": [
                                "search_strategy(query, format, limit)",
                                "search_rules(intent, limit)",
                                "search_card_corpus(query, format, limit)",
                            ],
                            "initial_candidate_cards": [
                                _document_to_model_context(document) for document in card_context[:24]
                            ],
                            "retrieved_rules_context": [
                                _document_to_model_context(document) for document in rules_context[:8]
                            ],
                            "retrieved_strategy_context": [
                                _document_to_model_context(document) for document in strategy_context[:10]
                            ],
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
        }
        logger.info("Ollama agent planning started", extra=log_extra(model=self.settings.ollama_model))
        response = await self._post_chat(payload)
        plan = _loads_json_content(response)
        logger.info(
            "Ollama agent planning completed",
            extra=log_extra(
                model=self.settings.ollama_model,
                card_query_count=len(plan.get("card_queries", [])),
            ),
        )
        return plan

    async def _run_tool_plan(
        self,
        request: DeckRequest,
        plan: dict[str, Any],
        steps: list[dict[str, str]],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "strategy": [],
            "rules": [],
            "card_corpus": [],
            "cards": [],
            "lookups": [],
        }

        for query in _string_list(plan.get("strategy_queries"))[:3]:
            documents = self.tools.search_strategy(query=query, mtg_format=request.format, limit=6)
            results["strategy"].extend(documents)
            steps.append(
                {
                    "label": "RAG strategy search",
                    "detail": f"{query} -> {len(documents)} documents",
                }
            )

        for query in _string_list(plan.get("rules_queries"))[:2]:
            documents = self.tools.search_rules(intent=query, limit=5)
            results["rules"].extend(documents)
            steps.append(
                {
                    "label": "RAG rules search",
                    "detail": f"{query} -> {len(documents)} documents",
                }
            )

        for query in _string_list(plan.get("card_queries"))[:5]:
            documents = self.tools.search_card_corpus(query=query, mtg_format=request.format, limit=24, request=request)
            results["card_corpus"].extend(documents)
            steps.append(
                {
                    "label": "RAG card corpus search",
                    "detail": f"{query} -> {len(documents)} card documents",
                }
            )

        return results

    async def _select_cards(
        self,
        request: DeckRequest,
        card_context: list[RetrievedDocument],
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
        tool_results: dict[str, Any],
        land_guidance: dict[str, int],
    ) -> dict[str, Any]:
        candidates = _candidate_payloads(card_context, tool_results, request=request, limit=45)
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "format": AGENT_SELECTION_SCHEMA,
            "options": {"temperature": 0.2},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Magic: The Gathering deck-building agent. Select a focused package "
                        "of cards from the provided candidates, including an appropriate mana base "
                        "when land candidates are available. The backend will apply copy counts and "
                        "validation. Use exact candidate names only. Return JSON. Respect deck size "
                        "and copy constraints: non-Commander constructed decks need at least 60 cards, "
                        "Commander decks need exactly 100 cards, non-basic cards are limited to four "
                        "copies in constructed, and Commander should use one copy of each non-basic. "
                        "Choose counts from 1 to 4 based on role: 4 for core cards, 2-3 for support "
                        "cards, and 1 for narrow, expensive, situational, or legendary cards. Value "
                        "budget_usd highly; if the user provides a budget, try to keep the final "
                        "estimated deck cost at or below that budget. When prices are present, prefer "
                        "lower estimated_price_usd options that still support the strategy, reduce "
                        "counts of expensive support cards, and avoid expensive mana bases unless the "
                        "budget can support them."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": request.model_dump(mode="json"),
                            "land_guidance": land_guidance,
                            "candidate_cards": candidates,
                            "retrieved_rules_context": [
                                _document_to_model_context(document) for document in rules_context[:6]
                            ],
                            "retrieved_strategy_context": [
                                _document_to_model_context(document) for document in strategy_context[:8]
                            ],
                            "instructions": (
                                "Pick 8 to 18 card names that best fit the request. Include useful "
                                "nonbasic lands when they fit the colors, format, and budget. Prefer "
                                "lower estimated prices when the request includes a budget. Return "
                                "selected_cards with exact name, count, and short role."
                            ),
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
        }
        logger.info(
            "Ollama card selection started",
            extra=log_extra(model=self.settings.ollama_model, candidate_count=len(candidates)),
        )
        response = await self._post_chat(payload)
        result = _loads_json_content(response)
        logger.info(
            "Ollama card selection completed",
            extra=log_extra(
                model=self.settings.ollama_model,
                selected_count=len(result.get("selected_cards", [])),
            ),
        )
        return result

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.settings.ollama_base_url, timeout=180) as client:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json()


def _loads_json_content(response_json: dict[str, Any]) -> dict[str, Any]:
    content = response_json.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama response did not include message content.")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(content[start : end + 1])


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _land_corpus_queries(request: DeckRequest) -> list[str]:
    color_names = {
        "W": "white",
        "U": "blue",
        "B": "black",
        "R": "red",
        "G": "green",
    }
    colors = [color.upper() for color in request.colors if color.upper() in color_names]
    color_text = " ".join([*colors, *(color_names[color] for color in colors)])
    budget_text = "cheap budget low price" if request.budget_usd is not None else "mana fixing"
    format_text = request.format.value
    queries = [
        f"{format_text} land {color_text} color identity mana fixing {budget_text}".strip(),
        f"{format_text} dual land fetch shock pain fast slow check land {color_text} {budget_text}".strip(),
    ]
    return list(dict.fromkeys(query for query in queries if query))


def _tool_card_names(tool_results: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for bucket in ("card_corpus", "cards", "lookups"):
        for item in tool_results.get(bucket, []):
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") or {}
            name = metadata.get("name") or item.get("title")
            if isinstance(name, str) and name and name not in names:
                names.append(name)
    return names


def _tool_card_payloads(tool_results: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bucket in ("lookups", "card_corpus", "cards"):
        for item in tool_results.get(bucket, []):
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") or {}
            name = metadata.get("name") or item.get("title")
            if not isinstance(name, str) or not name or name in seen:
                continue
            seen.add(name)
            payloads.append(item)
    return payloads


def _candidate_payloads(
    card_context: list[RetrievedDocument],
    tool_results: dict[str, Any],
    request: DeckRequest,
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_payload(payload: dict[str, Any]) -> None:
        metadata = payload.get("metadata") or {}
        name = metadata.get("name") or payload.get("title")
        if not isinstance(name, str) or not name or name in seen:
            return
        seen.add(name)
        candidates.append(
            {
                "name": name,
                "type_line": metadata.get("type_line"),
                "mana_value": metadata.get("mana_value"),
                "colors": metadata.get("colors"),
                "color_identity": metadata.get("color_identity"),
                "legalities": metadata.get("legalities"),
                "estimated_price_usd": metadata.get("estimated_price_usd"),
                "oracle_text": str(payload.get("content") or "")[:320],
                "content": str(payload.get("content") or "")[:320],
                "source_score": payload.get("score"),
            }
        )

    for document in card_context:
        add_payload(_document_to_model_context(document))

    for bucket in ("card_corpus", "lookups", "cards"):
        for payload in tool_results.get(bucket, []):
            if isinstance(payload, dict):
                add_payload(payload)

    return rank_candidate_cards(candidates, request)[:limit]


class OpenAIDeckAgent(AbstractDeckAgent):
    @property
    def thought_label(self) -> str:
        return "OpenAI thought"

    async def _plan(
        self,
        request: DeckRequest,
        card_context: list[RetrievedDocument],
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
    ) -> dict[str, Any]:
        payload = {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are a constrained Magic: The Gathering deck-building agent. "
                        "You already have initial RAG strategy and rules context. Plan extra RAG "
                        "tool calls for strategy, rules, and the embedded Scryfall card corpus. "
                        "Do not use live Scryfall for candidate discovery; the backend uses it "
                        "after deterministic validation to check current prices."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": request.model_dump(mode="json"),
                            "available_tools": [
                                "search_strategy(query, format, limit)",
                                "search_rules(intent, limit)",
                                "search_card_corpus(query, format, limit)",
                            ],
                            "initial_candidate_cards": [],
                            "retrieved_rules_context": [
                                _document_to_model_context(document) for document in rules_context[:8]
                            ],
                            "retrieved_strategy_context": [
                                _document_to_model_context(document) for document in strategy_context[:12]
                            ],
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
            "text": _json_schema_format("mtg_agent_plan", AGENT_PLAN_SCHEMA),
        }
        logger.info("OpenAI agent planning started", extra=log_extra(model=self.settings.openai_model))
        response = await self._post_response(payload)
        plan = _extract_json_response(response)
        logger.info(
            "OpenAI agent planning completed",
            extra=log_extra(
                model=self.settings.openai_model,
                card_query_count=len(plan.get("card_queries", [])),
            ),
        )
        return plan

    async def _run_tool_plan(self, request: DeckRequest, plan: dict[str, Any], steps: list[dict[str, str]]) -> dict[str, Any]:
        results: dict[str, Any] = {"strategy": [], "rules": [], "card_corpus": [], "cards": [], "lookups": []}

        for query in _string_list(plan.get("strategy_queries"))[:4]:
            documents = self.tools.search_strategy(query=query, mtg_format=request.format, limit=8)
            results["strategy"].extend(documents)
            steps.append({"label": "RAG strategy search", "detail": f"{query} -> {len(documents)} documents"})

        for query in _string_list(plan.get("rules_queries"))[:3]:
            documents = self.tools.search_rules(intent=query, limit=6)
            results["rules"].extend(documents)
            steps.append({"label": "RAG rules search", "detail": f"{query} -> {len(documents)} documents"})

        for query in _string_list(plan.get("card_queries"))[:6]:
            documents = self.tools.search_card_corpus(query=query, mtg_format=request.format, limit=28, request=request)
            results["card_corpus"].extend(documents)
            steps.append({"label": "RAG card corpus search", "detail": f"{query} -> {len(documents)} card documents"})

        for query in _land_corpus_queries(request):
            documents = self.tools.search_card_corpus(query=query, mtg_format=request.format, limit=16, request=request)
            results["card_corpus"].extend(documents)
            steps.append({"label": "RAG land corpus search", "detail": f"{query} -> {len(documents)} land documents"})

        return results

    async def _select_cards(
        self,
        request: DeckRequest,
        card_context: list[RetrievedDocument],
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
        tool_results: dict[str, Any],
        land_guidance: dict[str, int],
    ) -> dict[str, Any]:
        candidates = _candidate_payloads(card_context, tool_results, request=request, limit=80)
        payload = {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are a Magic: The Gathering deck-building agent. Select a coherent "
                        "package of cards from exact candidate names, including useful nonbasic lands "
                        "when land candidates are available. The backend will apply copy counts, fill "
                        "missing basic lands, and validate the final list. Use retrieved context for "
                        "strategy and rules. Return only the requested JSON. Respect deck size and "
                        "copy constraints: non-Commander constructed decks need at least 60 cards, "
                        "Commander decks need exactly 100 cards, non-basic cards are limited to four "
                        "copies in constructed, and Commander should use one copy of each non-basic. "
                        "Choose counts from 1 to 4 based on role: 4 for core cards, 2-3 for support "
                        "cards, and 1 for narrow, expensive, situational, or legendary cards. Value "
                        "budget_usd highly; if the user provides a budget, try to keep the final "
                        "estimated deck cost at or below that budget. When prices are present, prefer "
                        "lower estimated_price_usd options that still support the strategy, reduce "
                        "counts of expensive support cards, and avoid expensive mana bases unless the "
                        "budget can support them."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": request.model_dump(mode="json"),
                            "land_guidance": land_guidance,
                            "candidate_cards": candidates,
                            "retrieved_rules_context": [
                                _document_to_model_context(document) for document in rules_context[:12]
                            ],
                            "retrieved_strategy_context": [
                                _document_to_model_context(document) for document in strategy_context[:18]
                            ],
                            "agent_tool_results": tool_results,
                            "instructions": (
                                "Pick 8 to 22 cards. Use exact candidate names. Include mana-fixing "
                                "lands that fit the colors and format. If budget_usd is present, prefer "
                                "candidate lands and spells with lower estimated_price_usd. Explain the "
                                "deck's plan using retrieved strategy/rules and card corpus data. Return "
                                "selected_cards with exact name, count, and role."
                            ),
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
            "text": _json_schema_format("mtg_agent_selection", AGENT_SELECTION_SCHEMA),
        }
        logger.info(
            "OpenAI card selection started",
            extra=log_extra(model=self.settings.openai_model, candidate_count=len(candidates)),
        )
        response = await self._post_response(payload)
        result = _extract_json_response(response)
        logger.info(
            "OpenAI card selection completed",
            extra=log_extra(
                model=self.settings.openai_model,
                selected_count=len(result.get("selected_cards", [])),
            ),
        )
        return result

    async def _post_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.settings.openai_base_url,
            timeout=90,
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            response = await client.post("/responses", json=payload)
            response.raise_for_status()
            return response.json()


def _json_schema_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema,
        }
    }
