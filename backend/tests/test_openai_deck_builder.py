import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

from pydantic import BaseModel

from app.agent.deck_builder import (
    AGENT_RAG_PLAN_SCHEMA,
    AGENT_SCRYFALL_PLAN_SCHEMA,
    _aggregate_rag_context,
    _land_scryfall_queries,
)
from app.core.config import Settings
from app.core.openai_agents import run_structured_openai_agent
from app.models.deck import DeckRequest, Format
from app.rag.retriever import RetrievedDocument


class FakeDeckOutput(BaseModel):
    title: str


def test_run_structured_openai_agent_uses_agents_sdk(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeAgent:
        def __init__(
            self,
            *,
            name: str,
            instructions: str,
            model: str,
            output_type: type[BaseModel],
        ) -> None:
            self.name = name
            self.instructions = instructions
            self.model = model
            self.output_type = output_type
            calls["agent"] = self

    class FakeRunner:
        @staticmethod
        async def run(agent: FakeAgent, input: str) -> SimpleNamespace:
            calls["input"] = json.loads(input)
            return SimpleNamespace(final_output=agent.output_type(title="Deck"))

    fake_agents = ModuleType("agents")
    fake_agents.Agent = FakeAgent
    fake_agents.Runner = FakeRunner
    monkeypatch.setitem(sys.modules, "agents", fake_agents)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    result = asyncio.run(
        run_structured_openai_agent(
            settings=Settings(
                OPENAI_API_KEY="test-key",
                OPENAI_BASE_URL="https://example.test/v1",
                OPENAI_MODEL="gpt-test",
            ),
            name="MTG deck builder",
            instructions="Return JSON.",
            input_payload={"request": {"format": "modern"}},
            output_type=FakeDeckOutput,
        )
    )

    agent = calls["agent"]
    assert isinstance(agent, FakeAgent)
    assert agent.name == "MTG deck builder"
    assert agent.model == "gpt-test"
    assert calls["input"] == {"request": {"format": "modern"}}
    assert result == {"title": "Deck"}


def test_agent_plan_schemas_are_split_by_workflow_phase() -> None:
    assert "scryfall_queries" not in AGENT_RAG_PLAN_SCHEMA["required"]
    assert "scryfall_queries" not in AGENT_RAG_PLAN_SCHEMA["properties"]
    assert "strategy_queries" not in AGENT_SCRYFALL_PLAN_SCHEMA["required"]
    assert "strategy_queries" not in AGENT_SCRYFALL_PLAN_SCHEMA["properties"]


def test_land_scryfall_queries_use_live_search_filters() -> None:
    queries = _land_scryfall_queries(
        DeckRequest(format=Format.modern, colors=["U", "R"], budget_usd=80)
    )

    assert queries
    assert all("f:modern" in query for query in queries)
    assert all("id<=ur" in query for query in queries)
    assert all("t:land" in query for query in queries)
    assert all("usd<5" in query for query in queries)


def test_aggregate_rag_context_converts_tool_payload_dicts() -> None:
    strategy = [RetrievedDocument(title="Base", content="Core", source="mtgdecks_articles", metadata={})]
    rules = [RetrievedDocument(title="Rule 100.2", content="Decks", source="mtg_comprehensive_rules", metadata={})]
    rag_results = {
        "strategy": [
            {
                "title": "Izzet Prowess",
                "content": "Top deck cards: Monastery Swiftspear",
                "source": "mtgdecks_meta_decks",
                "metadata": {"format": "modern"},
                "score": 0.9,
            }
        ],
        "meta_decks": [
            {
                "title": "Ruby Storm (Modern)",
                "content": "Top deck cards: Ruby Medallion",
                "source": "mtgdecks_meta_decks",
                "metadata": {"format": "modern"},
                "score": 0.9,
            }
        ],
        "rules": [
            {
                "title": "Rule 601.2",
                "content": "Casting spells",
                "source": "mtg_comprehensive_rules",
                "metadata": {"rule_number": "601.2"},
            }
        ],
    }

    context = _aggregate_rag_context(strategy, rules, rag_results)

    assert [document.title for document in context["strategy"]] == ["Base", "Izzet Prowess"]
    assert [document.title for document in context["meta_decks"]] == ["Ruby Storm (Modern)"]
    assert [document.title for document in context["rules"]] == ["Rule 100.2", "Rule 601.2"]
