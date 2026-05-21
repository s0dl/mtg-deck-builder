from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.agent.context import document_to_model_context
from app.agent.tools import DeckAgentTools
from app.core.config import Settings
from app.core.logging import log_extra
from app.core.openai_agents import run_structured_openai_agent
from app.models.deck import DeckRequest
from app.rag.retriever import RetrievedDocument
from app.skills.deck_evaluation import rank_candidate_cards
from app.skills.deck_workflow import (
    request_constraints_instructions,
    request_constraints_payload,
    workflow_instructions,
    workflow_payload,
)

logger = logging.getLogger(__name__)

INITIAL_RULE_CONTEXT_LIMIT = 12
INITIAL_STRATEGY_CONTEXT_LIMIT = 24
PLANNED_STRATEGY_SEARCH_LIMIT = 40
PLANNED_META_DECK_SEARCH_LIMIT = 40
PLANNED_RULE_SEARCH_LIMIT = 20
SCRYFALL_PLAN_STRATEGY_CONTEXT_LIMIT = 30
SCRYFALL_PLAN_META_CONTEXT_LIMIT = 20
SELECTION_STRATEGY_CONTEXT_LIMIT = 24
SELECTION_META_CONTEXT_LIMIT = 16

AGENT_RAG_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["thoughts", "strategy_queries", "meta_deck_queries", "rules_queries"],
    "properties": {
        "thoughts": {"type": "array", "items": {"type": "string"}},
        "strategy_queries": {"type": "array", "items": {"type": "string"}},
        "meta_deck_queries": {"type": "array", "items": {"type": "string"}},
        "rules_queries": {"type": "array", "items": {"type": "string"}},
    },
}

AGENT_SCRYFALL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["thoughts", "scryfall_queries"],
    "properties": {
        "thoughts": {"type": "array", "items": {"type": "string"}},
        "scryfall_queries": {"type": "array", "items": {"type": "string"}},
    },
}

AGENT_SELECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "explanation", "selected_cards", "selected_lands"],
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
        "selected_lands": {
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


class AgentRagPlanOutput(BaseModel):
    thoughts: list[str] = Field(default_factory=list)
    strategy_queries: list[str] = Field(default_factory=list)
    meta_deck_queries: list[str] = Field(default_factory=list)
    rules_queries: list[str] = Field(default_factory=list)


class AgentScryfallPlanOutput(BaseModel):
    thoughts: list[str] = Field(default_factory=list)
    scryfall_queries: list[str] = Field(default_factory=list)


class AgentSelectedCardOutput(BaseModel):
    name: str
    count: int = Field(ge=1, le=4)
    role: str


class AgentSelectionOutput(BaseModel):
    title: str
    explanation: str
    selected_cards: list[AgentSelectedCardOutput] = Field(default_factory=list)
    selected_lands: list[AgentSelectedCardOutput] = Field(default_factory=list)


class AgenticDeckOutput(BaseModel):
    title: str
    explanation: str
    selected_cards: list[AgentSelectedCardOutput] = Field(default_factory=list)
    selected_lands: list[AgentSelectedCardOutput] = Field(default_factory=list)
    context_notes: list[str] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)


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
                "label": "Workflow skill loaded",
                "detail": (
                    "Using deck_builder_workflow: request constraints -> RAG tools -> corpus-backed card "
                    "tools -> candidate selection -> deterministic validation."
                ),
            },
            {
                "label": "GPT received initial RAG context",
                "detail": (
                    f"Loaded {len(strategy_context)} strategy documents, "
                    f"{len(rules_context)} rules documents, and {len(card_context)} initial live cards."
                ),
            }
        ]
        rag_plan = await self._plan_rag(request, rules_context, strategy_context)
        steps.append(
            {
                "label": "GPT RAG plan",
                "detail": (
                    f"Planned {len(_string_list(rag_plan.get('strategy_queries')))} strategy, "
                    f"{len(_string_list(rag_plan.get('meta_deck_queries')))} meta deck, and "
                    f"{len(_string_list(rag_plan.get('rules_queries')))} rules searches."
                ),
            }
        )
        for thought in _string_list(rag_plan.get("thoughts"))[:4]:
            steps.append({"label": self.thought_label, "detail": thought})

        rag_results = await self._run_rag_tool_plan(request, rag_plan, steps)
        rag_context = _aggregate_rag_context(strategy_context, rules_context, rag_results)

        scryfall_plan = await self._plan_scryfall(request, rag_context)
        steps.append(
            {
                "label": "GPT Scryfall plan",
                "detail": (
                    f"Planned {len(_string_list(scryfall_plan.get('scryfall_queries')))} corpus-backed card searches "
                    "from retrieved RAG context."
                ),
            }
        )
        for thought in _string_list(scryfall_plan.get("thoughts"))[:4]:
            steps.append({"label": self.thought_label, "detail": thought})

        tool_results = await self._run_scryfall_tool_plan(request, scryfall_plan, rag_context, steps, rag_results)
        result = await self._select_cards(
            request=request,
            card_context=card_context,
            rules_context=rag_context["rules"],
            strategy_context=rag_context["strategy"],
            tool_results=tool_results,
            land_guidance=land_guidance,
        )
        steps.append(
            {
                "label": "GPT card selection",
                "detail": (
                    f"Selected {len(result.get('selected_cards', []))} nonland card names and "
                    f"{len(result.get('selected_lands', []))} land names from corpus-backed candidates."
                ),
            }
        )
        result["agent_steps"] = steps
        result["tool_card_names"] = _tool_card_names(tool_results)
        result["tool_card_payloads"] = _tool_card_payloads(tool_results)
        result["tool_land_payloads"] = _tool_land_payloads(tool_results)
        result["tool_context_payloads"] = _tool_context_payloads(tool_results)
        return result

    @property
    @abstractmethod
    def thought_label(self) -> str:
        """Human-readable label for model reasoning snippets."""

    @abstractmethod
    async def _plan_rag(
        self,
        request: DeckRequest,
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
    ) -> dict[str, Any]:
        """Return a normalized RAG plan matching AGENT_RAG_PLAN_SCHEMA."""

    @abstractmethod
    async def _plan_scryfall(
        self,
        request: DeckRequest,
        rag_context: dict[str, list[RetrievedDocument]],
    ) -> dict[str, Any]:
        """Return a normalized corpus-backed card plan matching AGENT_SCRYFALL_PLAN_SCHEMA."""

    @abstractmethod
    async def _run_rag_tool_plan(
        self,
        request: DeckRequest,
        plan: dict[str, Any],
        steps: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Execute backend RAG tools requested by the plan and return normalized results."""

    @abstractmethod
    async def _run_scryfall_tool_plan(
        self,
        request: DeckRequest,
        plan: dict[str, Any],
        rag_context: dict[str, list[RetrievedDocument]],
        steps: list[dict[str, str]],
        rag_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute corpus-backed card tools requested by the plan and return normalized tool results."""

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

    async def _plan_rag(
        self,
        request: DeckRequest,
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
    ) -> dict[str, Any]:
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "format": AGENT_RAG_PLAN_SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        workflow_instructions("rag_planning")
                        + "\n\nYou are a constrained Magic: The Gathering deck-building agent. "
                        "First, plan only the extra RAG tool calls needed before deck construction. "
                        "Use strategy, meta-deck, and rules searches to gather context for the deck. "
                        "Do not plan any card corpus searches yet."
                        "\n\n"
                        + request_constraints_instructions(request)
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": request.model_dump(mode="json"),
                            "request_constraints": request_constraints_payload(request),
                            "deck_workflow": workflow_payload("rag_planning"),
                            "available_tools": self.tools.tool_signatures(
                                ("search_strategy", "search_meta_decks", "search_rules")
                            ),
                            "retrieved_rules_context": [
                                document_to_model_context(document)
                                for document in rules_context[:INITIAL_RULE_CONTEXT_LIMIT]
                            ],
                            "retrieved_strategy_context": [
                                document_to_model_context(document)
                                for document in strategy_context[:INITIAL_STRATEGY_CONTEXT_LIMIT]
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
                strategy_query_count=len(plan.get("strategy_queries", [])),
            ),
        )
        return plan

    async def _plan_scryfall(
        self,
        request: DeckRequest,
        rag_context: dict[str, list[RetrievedDocument]],
    ) -> dict[str, Any]:
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "format": AGENT_SCRYFALL_PLAN_SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        workflow_instructions("scryfall_planning")
                        + "\n\nYou are a constrained Magic: The Gathering deck-building agent. "
                        "You already have retrieved strategy, meta-deck, and rules context. "
                        "Now plan only card corpus searches to find cards that match the retrieved "
                        "documents. Use the documents at hand to name relevant archetype pieces, "
                        "staples, mana bases, and support cards."
                        "\n\n"
                        + request_constraints_instructions(request)
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": request.model_dump(mode="json"),
                            "request_constraints": request_constraints_payload(request),
                            "deck_workflow": workflow_payload("scryfall_planning"),
                            "available_tools": self.tools.tool_signatures(
                                ("search_cards_scryfall", "search_card_corpus")
                            ),
                            "retrieved_strategy_context": [
                                document_to_model_context(document)
                                for document in rag_context["strategy"][:SCRYFALL_PLAN_STRATEGY_CONTEXT_LIMIT]
                            ],
                            "retrieved_meta_deck_context": [
                                document_to_model_context(document)
                                for document in rag_context["meta_decks"][:SCRYFALL_PLAN_META_CONTEXT_LIMIT]
                            ],
                            "retrieved_rules_context": [
                                document_to_model_context(document)
                                for document in rag_context["rules"][:INITIAL_RULE_CONTEXT_LIMIT]
                            ],
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
        }
        logger.info("Ollama corpus-backed card planning started", extra=log_extra(model=self.settings.ollama_model))
        response = await self._post_chat(payload)
        plan = _loads_json_content(response)
        logger.info(
            "Ollama corpus-backed card planning completed",
            extra=log_extra(
                model=self.settings.ollama_model,
                scryfall_query_count=len(plan.get("scryfall_queries", [])),
            ),
        )
        return plan

    async def _run_rag_tool_plan(
        self,
        request: DeckRequest,
        plan: dict[str, Any],
        steps: list[dict[str, str]],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "strategy": [],
            "meta_decks": [],
            "rules": [],
            "cards": [],
            "lookups": [],
        }

        for query in _string_list(plan.get("strategy_queries"))[:3]:
            documents = self.tools.search_strategy(
                query=query,
                mtg_format=request.format,
                limit=PLANNED_STRATEGY_SEARCH_LIMIT,
            )
            results["strategy"].extend(documents)
            steps.append(
                {
                    "label": "RAG strategy search",
                    "detail": f"{query} -> {len(documents)} documents",
                }
            )

        for query in _string_list(plan.get("meta_deck_queries"))[:3]:
            documents = self.tools.search_meta_decks(
                query=query,
                mtg_format=request.format,
                limit=PLANNED_META_DECK_SEARCH_LIMIT,
            )
            results["meta_decks"].extend(documents)
            steps.append(
                {
                    "label": "RAG meta deck search",
                    "detail": f"{query} -> {len(documents)} meta deck documents",
                }
            )

        for query in _string_list(plan.get("rules_queries"))[:2]:
            documents = self.tools.search_rules(intent=query, limit=PLANNED_RULE_SEARCH_LIMIT)
            results["rules"].extend(documents)
            steps.append(
                {
                    "label": "RAG rules search",
                    "detail": f"{query} -> {len(documents)} documents",
                }
            )

        return results

    async def _run_scryfall_tool_plan(
        self,
        request: DeckRequest,
        plan: dict[str, Any],
        rag_context: dict[str, list[RetrievedDocument]],
        steps: list[dict[str, str]],
        rag_results: dict[str, Any],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "strategy": rag_results["strategy"],
            "meta_decks": rag_results["meta_decks"],
            "rules": rag_results["rules"],
            "cards": [],
            "lands": [],
            "lookups": [],
        }

        for name in request.must_include:
            try:
                document = await self.tools.lookup_card(name=name)
            except httpx.HTTPError as exc:
                steps.append({"label": "Requested card lookup failed", "detail": f"{name} -> {exc}"})
                continue
            results["lookups"].append(document)
            steps.append({"label": "Requested card lookup", "detail": f"{name} -> corpus-backed card candidate"})

        for query in _string_list(plan.get("scryfall_queries"))[:6]:
            try:
                documents = await self.tools.search_cards_scryfall(query=query, limit=20)
            except httpx.HTTPError as exc:
                steps.append({"label": "Corpus-backed card search failed", "detail": f"{query} -> {exc}"})
                continue
            results["cards"].extend(documents)
            steps.append(
                {
                    "label": "Live Scryfall search",
                    "detail": f"{query} -> {len(documents)} live cards",
                }
            )

        for query in _land_scryfall_queries(request):
            try:
                documents = self.tools.search_card_corpus(
                    query=query,
                    mtg_format=request.format,
                    request=request,
                    limit=16,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                steps.append({"label": "Corpus-backed land search failed", "detail": f"{query} -> {exc}"})
                continue
            results["lands"].extend(documents)
            steps.append(
                {
                    "label": "Corpus-backed land search",
                    "detail": f"{query} -> {len(documents)} lands",
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
        land_candidates = _candidate_payloads(
            card_context,
            tool_results,
            request=request,
            limit=24,
            include_lands=True,
        )
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "format": AGENT_SELECTION_SCHEMA,
            "options": {"temperature": 0.2},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        workflow_instructions("card_selection")
                        + "\n\nYou are a Magic: The Gathering deck-building agent. Select a focused package "
                        "of nonland cards and a separate mana base from the provided candidates. "
                        "Return nonland spells in selected_cards and lands in selected_lands. "
                        "The backend will apply copy counts, fill missing land slots, and "
                        "validation. Use exact candidate names only. Return JSON. Respect deck size "
                        "and copy constraints: non-Commander constructed decks need at least 60 cards, "
                        "Commander decks need exactly 100 cards, non-basic cards are limited to four "
                        "copies in constructed, and Commander should use one copy of each non-basic. "
                        "Choose counts from 1 to 4 based on role: 4 for core cards, 2-3 for support "
                        "cards, and 1 for narrow, expensive, situational, or legendary cards. Treat "
                        "budget_usd as a ceiling for power, not as a request for the cheapest possible "
                        "deck. For high budgets, prefer meta-proven staples and premium mana bases if "
                        "they improve the deck while staying under budget."
                        "\n\n"
                        + request_constraints_instructions(request)
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": request.model_dump(mode="json"),
                            "request_constraints": request_constraints_payload(request),
                            "deck_workflow": workflow_payload("card_selection"),
                            "budget_guidance": _budget_guidance(request),
                            "land_guidance": land_guidance,
                            "candidate_cards": candidates,
                            "land_candidates": land_candidates,
                            "retrieved_rules_context": [
                                document_to_model_context(document)
                                for document in rules_context[:INITIAL_RULE_CONTEXT_LIMIT]
                            ],
                            "retrieved_strategy_context": [
                                document_to_model_context(document)
                                for document in strategy_context[:SELECTION_STRATEGY_CONTEXT_LIMIT]
                            ],
                            "retrieved_meta_deck_context": [
                                payload
                                for payload in tool_results.get("meta_decks", [])[:SELECTION_META_CONTEXT_LIMIT]
                                if isinstance(payload, dict)
                            ],
                            "instructions": (
                                "Pick enough nonland card names to fill the nonland portion of the "
                                "deck after lands, usually 9 to 16 names with appropriate counts. "
                                "Include useful nonbasic lands in selected_lands when they fit the "
                                "colors, format, and budget. Use more "
                                "of a large budget for stronger staples instead of defaulting to the "
                                "cheapest legal cards. The backend will not choose unselected spells. Return "
                                "selected_cards and selected_lands with exact name, count, and short role."
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


def _loads_json_array(value: str, label: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Agent tool expected {label} as a JSON array.") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Agent tool expected {label} as a JSON array.")
    return [item for item in payload if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _land_scryfall_queries(request: DeckRequest) -> list[str]:
    colors = [color.upper() for color in request.colors if color.upper() in {"W", "U", "B", "R", "G"}]
    price_filter = _land_price_filter(request)
    format_text = f"f:{request.format.value}" if request.format.value != "casual" else ""
    color_identity = "".join(color.lower() for color in "WUBRG" if color in colors)
    color_text = f"id<={color_identity}" if color_identity else ""
    queries = [
        f"{format_text} {color_text} game:paper -is:digital t:land {price_filter}".strip(),
        (
            f"{format_text} {color_text} game:paper -is:digital t:land "
            f"(t:mountain or t:island or t:swamp or t:forest or t:plains or o:add) {price_filter}"
        ).strip(),
    ]
    return list(dict.fromkeys(query for query in queries if query))


def _tool_card_names(tool_results: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for bucket in ("cards", "lands", "lookups"):
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
    for bucket in ("lookups", "cards"):
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


def _tool_land_payloads(tool_results: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in tool_results.get("lands", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        name = metadata.get("name") or item.get("title")
        if not isinstance(name, str) or not name or name in seen:
            continue
        seen.add(name)
        payloads.append(item)
    return payloads


def _tool_context_payloads(tool_results: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for bucket in ("meta_decks", "strategy", "rules"):
        for item in tool_results.get(bucket, []):
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            source = item.get("source")
            if not isinstance(title, str) or not isinstance(source, str):
                continue
            key = (source, title)
            if key in seen:
                continue
            seen.add(key)
            payloads.append(item)
    return payloads


def _budget_guidance(request: DeckRequest) -> str:
    if request.budget_usd is None:
        return "No budget was provided; optimize for card quality and synergy."
    if request.budget_usd >= 300:
        return (
            f"Budget is ${request.budget_usd:.0f}. This is a high ceiling: optimize for power, "
            "meta relevance, and premium mana while staying under budget. Do not choose budget "
            "substitutes just because they are cheaper."
        )
    if request.budget_usd >= 100:
        return (
            f"Budget is ${request.budget_usd:.0f}. Use efficient staples and solid mana, but avoid "
            "luxury upgrades that crowd out core cards."
        )
    return (
        f"Budget is ${request.budget_usd:.0f}. This is a tight budget: prefer low-price cards and "
        "avoid expensive lands unless they are essential."
    )


def _land_price_filter(request: DeckRequest) -> str:
    if request.budget_usd is None or request.budget_usd >= 300:
        return ""
    if request.budget_usd >= 100:
        return "usd<25"
    return "usd<5"


def _aggregate_rag_context(
    strategy_context: list[RetrievedDocument],
    rules_context: list[RetrievedDocument],
    rag_results: dict[str, Any],
) -> dict[str, list[RetrievedDocument]]:
    strategy = _dedupe_retrieved_documents(
        [
            *strategy_context,
            *[
                document
                for payload in rag_results.get("strategy", [])
                if (document := _payload_to_retrieved_document(payload)) is not None
            ],
        ]
    )
    meta_decks = _dedupe_retrieved_documents(
        [
            *[
                document
                for payload in rag_results.get("meta_decks", [])
                if (document := _payload_to_retrieved_document(payload)) is not None
            ]
        ]
    )
    rules = _dedupe_retrieved_documents(
        [
            *rules_context,
            *[
                document
                for payload in rag_results.get("rules", [])
                if (document := _payload_to_retrieved_document(payload)) is not None
            ],
        ]
    )
    return {
        "strategy": strategy,
        "meta_decks": meta_decks,
        "rules": rules,
    }


def _dedupe_retrieved_documents(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    deduped: list[RetrievedDocument] = []
    seen: set[tuple[str, str, str]] = set()
    for document in documents:
        key = (document.source, document.title, document.content[:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(document)
    return deduped


def _payload_to_retrieved_document(payload: Any) -> RetrievedDocument | None:
    if not isinstance(payload, dict):
        return None
    title = payload.get("title")
    content = payload.get("content")
    source = payload.get("source")
    metadata = payload.get("metadata")
    if not isinstance(title, str) or not isinstance(content, str) or not isinstance(source, str):
        return None
    if not isinstance(metadata, dict):
        metadata = {}
    score = payload.get("score")
    return RetrievedDocument(
        title=title,
        content=content,
        source=source,
        metadata=metadata,
        score=float(score) if isinstance(score, (int, float)) else None,
    )


def _payload_is_land(payload: dict[str, Any]) -> bool:
    metadata = payload.get("metadata") or {}
    type_line = str(metadata.get("type_line") or "")
    return "land" in {part.lower() for part in type_line.split(" ")}


def _candidate_payloads(
    card_context: list[RetrievedDocument],
    tool_results: dict[str, Any],
    request: DeckRequest,
    limit: int,
    *,
    include_lands: bool = False,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_payload(payload: dict[str, Any]) -> None:
        if _payload_is_land(payload) != include_lands:
            return
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
        add_payload(document_to_model_context(document))

    for bucket in ("lookups", "cards", "lands"):
        for payload in tool_results.get(bucket, []):
            if isinstance(payload, dict):
                add_payload(payload)

    return rank_candidate_cards(candidates, request)[:limit]


class OpenAIDeckAgent(AbstractDeckAgent):
    @property
    def thought_label(self) -> str:
        return "OpenAI thought"

    async def generate(
        self,
        request: DeckRequest,
        card_context: list[RetrievedDocument],
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
        land_guidance: dict[str, int],
    ) -> dict[str, Any]:
        try:
            from agents import function_tool
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Agents SDK is required for OpenAI generation. "
                "Install backend dependencies so the 'openai-agents' package is available."
            ) from exc

        steps: list[dict[str, str]] = [
            {
                "label": "OpenAI manager agent",
                "detail": (
                    "Manager owns the RAG, Scryfall, candidate evaluation, and validation tool loop; "
                    "deterministic backend finalization still enforces deck size, lands, copy limits, and legality."
                ),
            },
            {
                "label": "Initial RAG context",
                "detail": (
                    f"Seeded manager with {len(strategy_context)} strategy documents, "
                    f"{len(rules_context)} rules documents, and {len(card_context)} initial live cards."
                ),
            },
        ]
        tool_results: dict[str, Any] = {
            "strategy": [],
            "meta_decks": [],
            "rules": [],
            "cards": [],
            "lands": [],
            "lookups": [],
            "evaluations": [],
            "validations": [],
        }

        for query in _land_scryfall_queries(request):
            try:
                documents = self.tools.search_card_corpus(
                    query=query,
                    mtg_format=request.format,
                    request=request,
                    limit=24,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                steps.append({"label": "Dedicated land corpus search failed", "detail": f"{query} -> {exc}"})
                continue
            tool_results["lands"].extend(documents)
            steps.append({"label": "Dedicated land corpus search", "detail": f"{query} -> {len(documents)} lands"})

        @function_tool
        def search_strategy(query: str, limit: int = 40) -> list[dict[str, Any]]:
            """Search strategy articles and nearby meta-deck context for this request's format."""
            documents = self.tools.search_strategy(query=query, mtg_format=request.format, limit=limit)
            tool_results["strategy"].extend(documents)
            steps.append({"label": "Agent strategy search", "detail": f"{query} -> {len(documents)} documents"})
            return documents

        @function_tool
        def search_meta_decks(query: str, limit: int = 40) -> list[dict[str, Any]]:
            """Search ingested tournament/meta decklists for archetype and card-package evidence."""
            documents = self.tools.search_meta_decks(query=query, mtg_format=request.format, limit=limit)
            tool_results["meta_decks"].extend(documents)
            steps.append({"label": "Agent meta deck search", "detail": f"{query} -> {len(documents)} documents"})
            return documents

        @function_tool
        def search_rules(intent: str, limit: int = 20) -> list[dict[str, Any]]:
            """Search comprehensive rules context for a specific deck-building or legality question."""
            documents = self.tools.search_rules(intent=intent, limit=limit)
            tool_results["rules"].extend(documents)
            steps.append({"label": "Agent rules search", "detail": f"{intent} -> {len(documents)} documents"})
            return documents

        @function_tool
        async def search_cards_scryfall(query: str, limit: int = 24) -> list[dict[str, Any]]:
            """Search the indexed card corpus for exact paper card candidates."""
            documents = await self.tools.search_cards_scryfall(query=query, limit=limit)
            tool_results["cards"].extend(documents)
            steps.append({"label": "Agent card search", "detail": f"{query} -> {len(documents)} cards"})
            return documents

        @function_tool
        def search_card_corpus(query: str, limit: int = 24) -> list[dict[str, Any]]:
            """Search the indexed card corpus with this deck request's format, color, budget, and land filters."""
            documents = self.tools.search_card_corpus(
                query=query,
                mtg_format=request.format,
                request=request,
                limit=limit,
            )
            bucket = "lands" if any(_payload_is_land(document) for document in documents) else "cards"
            tool_results[bucket].extend(documents)
            steps.append({"label": "Agent corpus search", "detail": f"{query} -> {len(documents)} cards"})
            return documents

        @function_tool
        async def lookup_card(name: str) -> dict[str, Any]:
            """Look up a requested or exact card name in the indexed card corpus."""
            document = await self.tools.lookup_card(name=name)
            tool_results["lookups"].append(document)
            steps.append({"label": "Agent card lookup", "detail": f"{name} -> corpus-backed card"})
            return document

        @function_tool
        def curate_context(max_notes: int = 15) -> list[str]:
            """Compress gathered RAG context into a short list of high-signal notes."""
            documents = [
                *[document_to_model_context(document) for document in strategy_context],
                *[document_to_model_context(document) for document in rules_context],
                *tool_results["strategy"],
                *tool_results["meta_decks"],
                *tool_results["rules"],
            ]
            notes = self.tools.curate_context_notes(documents, max_notes=max_notes)
            steps.append({"label": "Agent context curation", "detail": f"Curated {len(notes)} notes from gathered context"})
            return notes

        @function_tool
        def evaluate_deck_candidates(cards_json: str) -> list[dict[str, Any]]:
            """Score JSON-encoded candidate card payloads against the request's constraints."""
            cards = _loads_json_array(cards_json, "candidate cards")
            evaluations = self.tools.evaluate_deck_candidates(cards, request)
            tool_results["evaluations"].append(evaluations)
            steps.append({"label": "Agent candidate evaluation", "detail": f"Evaluated {len(evaluations)} candidates"})
            return evaluations

        @function_tool
        def validate_deck_cards(cards_json: str) -> dict[str, Any]:
            """Validate a JSON-encoded complete decklist against deterministic deck construction rules."""
            cards = _loads_json_array(cards_json, "deck cards")
            validation = self.tools.validate_deck_cards(cards, request.format)
            tool_results["validations"].append(validation)
            status = "valid" if validation.get("is_valid") else "invalid"
            error_count = len(validation.get("errors") or [])
            steps.append({"label": "Agent validation", "detail": f"Proposed deck was {status} with {error_count} errors"})
            return validation

        instructions = (
            "You are DeckManagerAgent for Magic: The Gathering deck construction. "
            "Own the workflow: plan searches, call tools, inspect results, revise queries, curate context, "
            "evaluate candidate cards, and only then return selected_cards. Use exact names from Scryfall "
            "or lookup_card results. Call validate_deck_cards when you have a complete proposed list; if it "
            "fails, revise before your final answer. The backend will deterministically finalize land "
            "counts, deck size, copy limits, budget, and legality after your output, but it will not "
            "choose unselected spells. "
            "Make mana base construction explicit: use land_candidates and search_card_corpus land queries, "
            "then return nonland cards in selected_cards and lands in selected_lands. "
            "Selected_cards must cover the deck's nonland slots: prefer 9 to 16 high-quality nonland "
            "selections with counts reflecting role, plus a separate nonbasic-land package. "
            "Use curate_context after broad retrieval so the final explanation is based on compact notes."
            "\n\n"
            + request_constraints_instructions(request)
        )
        input_payload = {
            "request": request.model_dump(mode="json"),
            "request_constraints": request_constraints_payload(request),
            "budget_guidance": _budget_guidance(request),
            "land_guidance": land_guidance,
            "land_candidates": _candidate_payloads(
                card_context,
                tool_results,
                request=request,
                limit=32,
                include_lands=True,
            ),
            "initial_strategy_context": [
                document_to_model_context(document)
                for document in strategy_context[:INITIAL_STRATEGY_CONTEXT_LIMIT]
            ],
            "initial_rules_context": [
                document_to_model_context(document) for document in rules_context[:INITIAL_RULE_CONTEXT_LIMIT]
            ],
            "available_tool_names": [
                "search_strategy",
                "search_meta_decks",
                "search_rules",
                "search_cards_scryfall",
                "search_card_corpus",
                "lookup_card",
                "curate_context",
                "evaluate_deck_candidates",
                "validate_deck_cards",
            ],
        }
        result = await run_structured_openai_agent(
            settings=self.settings,
            name="MTG deck manager",
            instructions=instructions,
            input_payload=input_payload,
            output_type=AgenticDeckOutput,
            tools=[
                search_strategy,
                search_meta_decks,
                search_rules,
                search_cards_scryfall,
                search_card_corpus,
                lookup_card,
                curate_context,
                evaluate_deck_candidates,
                validate_deck_cards,
            ],
            max_turns=12,
        )
        steps.append(
            {
                "label": "Agent final selection",
                "detail": (
                    f"Returned {len(result.get('selected_cards', []))} nonland card names and "
                    f"{len(result.get('selected_lands', []))} land names after tool-driven planning."
                ),
            }
        )
        result["agent_steps"] = steps
        result["tool_card_names"] = _tool_card_names(tool_results)
        result["tool_card_payloads"] = _tool_card_payloads(tool_results)
        result["tool_land_payloads"] = _tool_land_payloads(tool_results)
        result["tool_context_payloads"] = _tool_context_payloads(tool_results)
        return result

    async def _plan_rag(
        self,
        request: DeckRequest,
        rules_context: list[RetrievedDocument],
        strategy_context: list[RetrievedDocument],
    ) -> dict[str, Any]:
        instructions = (
            workflow_instructions("rag_planning")
            + "\n\nYou are a constrained Magic: The Gathering deck-building agent. "
            "First, plan only the extra RAG tool calls needed before deck construction. "
            "Use strategy, meta-deck, and rules searches to gather context for the deck. "
            "Do not plan any card corpus searches yet."
            "\n\n"
            + request_constraints_instructions(request)
        )
        input_payload = {
            "request": request.model_dump(mode="json"),
            "request_constraints": request_constraints_payload(request),
            "deck_workflow": workflow_payload("rag_planning"),
            "available_tools": self.tools.tool_signatures(
                ("search_strategy", "search_meta_decks", "search_rules")
            ),
            "retrieved_rules_context": [
                document_to_model_context(document) for document in rules_context[:INITIAL_RULE_CONTEXT_LIMIT]
            ],
            "retrieved_strategy_context": [
                document_to_model_context(document)
                for document in strategy_context[:INITIAL_STRATEGY_CONTEXT_LIMIT]
            ],
        }
        logger.info("OpenAI RAG planning started", extra=log_extra(model=self.settings.openai_model))
        plan = await run_structured_openai_agent(
            settings=self.settings,
            name="MTG RAG planner",
            instructions=instructions,
            input_payload=input_payload,
            output_type=AgentRagPlanOutput,
        )
        logger.info(
            "OpenAI RAG planning completed",
            extra=log_extra(
                model=self.settings.openai_model,
                strategy_query_count=len(plan.get("strategy_queries", [])),
            ),
        )
        return plan

    async def _plan_scryfall(
        self,
        request: DeckRequest,
        rag_context: dict[str, list[RetrievedDocument]],
    ) -> dict[str, Any]:
        instructions = (
            workflow_instructions("scryfall_planning")
            + "\n\nYou are a constrained Magic: The Gathering deck-building agent. "
            "You already have retrieved strategy, meta-deck, and rules context. "
            "Now plan only card corpus searches to find cards that match the retrieved "
            "documents. Use the documents at hand to name relevant archetype pieces, "
            "staples, mana bases, and support cards."
            "\n\n"
            + request_constraints_instructions(request)
        )
        input_payload = {
            "request": request.model_dump(mode="json"),
            "request_constraints": request_constraints_payload(request),
            "deck_workflow": workflow_payload("scryfall_planning"),
            "available_tools": self.tools.tool_signatures(("search_cards_scryfall", "search_card_corpus")),
            "retrieved_strategy_context": [
                document_to_model_context(document)
                for document in rag_context["strategy"][:SCRYFALL_PLAN_STRATEGY_CONTEXT_LIMIT]
            ],
            "retrieved_meta_deck_context": [
                document_to_model_context(document)
                for document in rag_context["meta_decks"][:SCRYFALL_PLAN_META_CONTEXT_LIMIT]
            ],
            "retrieved_rules_context": [
                document_to_model_context(document)
                for document in rag_context["rules"][:INITIAL_RULE_CONTEXT_LIMIT]
            ],
        }
        logger.info("OpenAI corpus-backed card planning started", extra=log_extra(model=self.settings.openai_model))
        plan = await run_structured_openai_agent(
            settings=self.settings,
            name="MTG Scryfall planner",
            instructions=instructions,
            input_payload=input_payload,
            output_type=AgentScryfallPlanOutput,
        )
        logger.info(
            "OpenAI corpus-backed card planning completed",
            extra=log_extra(
                model=self.settings.openai_model,
                scryfall_query_count=len(plan.get("scryfall_queries", [])),
            ),
        )
        return plan

    async def _run_rag_tool_plan(
        self,
        request: DeckRequest,
        plan: dict[str, Any],
        steps: list[dict[str, str]],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {"strategy": [], "meta_decks": [], "rules": []}

        for query in _string_list(plan.get("strategy_queries"))[:4]:
            documents = self.tools.search_strategy(
                query=query,
                mtg_format=request.format,
                limit=PLANNED_STRATEGY_SEARCH_LIMIT,
            )
            results["strategy"].extend(documents)
            steps.append({"label": "RAG strategy search", "detail": f"{query} -> {len(documents)} documents"})

        for query in _string_list(plan.get("meta_deck_queries"))[:4]:
            documents = self.tools.search_meta_decks(
                query=query,
                mtg_format=request.format,
                limit=PLANNED_META_DECK_SEARCH_LIMIT,
            )
            results["meta_decks"].extend(documents)
            steps.append({"label": "RAG meta deck search", "detail": f"{query} -> {len(documents)} meta deck documents"})

        for query in _string_list(plan.get("rules_queries"))[:3]:
            documents = self.tools.search_rules(intent=query, limit=PLANNED_RULE_SEARCH_LIMIT)
            results["rules"].extend(documents)
            steps.append({"label": "RAG rules search", "detail": f"{query} -> {len(documents)} documents"})

        return results

    async def _run_scryfall_tool_plan(
        self,
        request: DeckRequest,
        plan: dict[str, Any],
        rag_context: dict[str, list[RetrievedDocument]],
        steps: list[dict[str, str]],
        rag_results: dict[str, Any],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "strategy": rag_results["strategy"],
            "meta_decks": rag_results["meta_decks"],
            "rules": rag_results["rules"],
            "cards": [],
            "lands": [],
            "lookups": [],
        }

        for name in request.must_include:
            try:
                document = await self.tools.lookup_card(name=name)
            except httpx.HTTPError as exc:
                steps.append({"label": "Requested card lookup failed", "detail": f"{name} -> {exc}"})
                continue
            results["lookups"].append(document)
            steps.append({"label": "Requested card lookup", "detail": f"{name} -> corpus-backed card candidate"})

        for query in _string_list(plan.get("scryfall_queries"))[:8]:
            try:
                documents = await self.tools.search_cards_scryfall(query=query, limit=24)
            except httpx.HTTPError as exc:
                steps.append({"label": "Live Scryfall search failed", "detail": f"{query} -> {exc}"})
                continue
            results["cards"].extend(documents)
            steps.append({"label": "Corpus-backed card search", "detail": f"{query} -> {len(documents)} cards"})

        for query in _land_scryfall_queries(request):
            try:
                documents = self.tools.search_card_corpus(
                    query=query,
                    mtg_format=request.format,
                    request=request,
                    limit=20,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                steps.append({"label": "Corpus-backed land search failed", "detail": f"{query} -> {exc}"})
                continue
            results["lands"].extend(documents)
            steps.append({"label": "Corpus-backed land search", "detail": f"{query} -> {len(documents)} lands"})

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
        land_candidates = _candidate_payloads(
            card_context,
            tool_results,
            request=request,
            limit=32,
            include_lands=True,
        )
        instructions = (
            workflow_instructions("card_selection")
            + "\n\nYou are a Magic: The Gathering deck-building agent. Select a coherent "
            "package of nonland cards from exact candidate names, and select useful "
            "nonbasic lands separately in selected_lands when land candidates are available. "
            "The backend will apply copy counts, fill missing land slots, and validate the final list; "
            "it will not choose unselected spells. Use retrieved context for "
            "strategy and rules. Respect deck size and copy constraints: non-Commander "
            "constructed decks need at least 60 cards, Commander decks need exactly 100 "
            "cards, non-basic cards are limited to four copies in constructed, and Commander "
            "should use one copy of each non-basic. Choose counts from 1 to 4 based on role: "
            "4 for core cards, 2-3 for support cards, and 1 for narrow, expensive, "
            "situational, or legendary cards. Treat budget_usd as a ceiling for power, not "
            "as a request for the cheapest possible deck. For high budgets, prefer "
            "meta-proven staples and premium mana bases if they improve the deck while "
            "staying under budget."
            "\n\n"
            + request_constraints_instructions(request)
        )
        input_payload = {
            "request": request.model_dump(mode="json"),
            "request_constraints": request_constraints_payload(request),
            "deck_workflow": workflow_payload("card_selection"),
            "budget_guidance": _budget_guidance(request),
            "land_guidance": land_guidance,
            "candidate_cards": candidates,
            "land_candidates": land_candidates,
            "retrieved_rules_context": [
                document_to_model_context(document) for document in rules_context[:INITIAL_RULE_CONTEXT_LIMIT]
            ],
            "retrieved_strategy_context": [
                document_to_model_context(document)
                for document in strategy_context[:SELECTION_STRATEGY_CONTEXT_LIMIT]
            ],
            "retrieved_meta_deck_context": [
                payload
                for payload in tool_results.get("meta_decks", [])[:SELECTION_META_CONTEXT_LIMIT]
                if isinstance(payload, dict)
            ],
            "agent_tool_results": tool_results,
            "instructions": (
                "Pick enough nonland cards to fill the nonland portion of the deck after lands, "
                "usually 9 to 16 names with appropriate counts, and a separate selected_lands package. "
                "Use exact candidate names. Include mana-fixing lands that fit the "
                "colors and format. If budget_usd is high, use it "
                "for stronger staples and a better mana base instead of defaulting to "
                "the cheapest legal candidates. Explain the deck's plan using retrieved "
                "strategy/rules and corpus-backed card data."
            ),
        }
        logger.info(
            "OpenAI card selection started",
            extra=log_extra(model=self.settings.openai_model, candidate_count=len(candidates)),
        )
        result = await run_structured_openai_agent(
            settings=self.settings,
            name="MTG card selector",
            instructions=instructions,
            input_payload=input_payload,
            output_type=AgentSelectionOutput,
        )
        logger.info(
            "OpenAI card selection completed",
            extra=log_extra(
                model=self.settings.openai_model,
                selected_count=len(result.get("selected_cards", [])),
            ),
        )
        return result
