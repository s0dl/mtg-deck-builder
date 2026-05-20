import asyncio
from typing import Any

from app.agent.deck_builder import OpenAIDeckAgent
from app.core.config import Settings
from app.models.deck import DeckRequest, Format
from app.skills.deck_workflow import (
    PHASE_ALLOWED_TOOLS,
    request_constraints_payload,
    workflow_instructions,
    workflow_payload,
    workflow_tool_order,
)


class FakeTools:
    def tool_signatures(self, names: tuple[str, ...] | list[str]) -> list[str]:
        return [f"{name}()" for name in names]

    async def lookup_card(self, name: str) -> dict[str, Any]:
        return {
            "title": name,
            "content": f"{name} card text",
            "source": "scryfall_live",
            "metadata": {"name": name, "type_line": "Creature", "legalities": {"modern": "legal"}},
        }

    async def search_cards_scryfall(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return []


def test_workflow_orders_rag_before_live_scryfall_and_validation() -> None:
    tool_order = workflow_tool_order()

    assert tool_order.index("search_strategy") < tool_order.index("search_cards_scryfall")
    assert tool_order.index("search_meta_decks") < tool_order.index("search_cards_scryfall")
    assert tool_order.index("search_rules") < tool_order.index("search_cards_scryfall")
    assert tool_order.index("search_cards_scryfall") < tool_order.index("validate_deck_cards")


def test_workflow_payload_limits_tools_to_current_phase() -> None:
    rag_payload = workflow_payload("rag_planning")
    scryfall_payload = workflow_payload("scryfall_planning")
    selection_payload = workflow_payload("card_selection")

    assert rag_payload["phase_allowed_tools"] == list(PHASE_ALLOWED_TOOLS["rag_planning"])
    assert "search_cards_scryfall" not in rag_payload["phase_allowed_tools"]
    assert scryfall_payload["phase_allowed_tools"] == ["search_cards_scryfall"]
    assert selection_payload["phase_allowed_tools"] == []


def test_workflow_instructions_bind_the_model_to_one_phase() -> None:
    instructions = workflow_instructions("scryfall_planning")

    assert "Current phase: scryfall_planning." in instructions
    assert "Plan only live Scryfall searches" in instructions
    assert "Do not ask for tools outside the current workflow phase." in instructions


def test_rag_planning_receives_comprehensive_rules_index() -> None:
    payload = workflow_payload("rag_planning")
    instructions = workflow_instructions("rag_planning")

    assert "mtg_comprehensive_rules_index" in payload
    assert payload["mtg_comprehensive_rules_index"][0]["section"] == "1. Game Concepts"
    assert "117. Timing and Priority" in payload["mtg_comprehensive_rules_index"][0]["rules"]
    assert "903. Commander" in payload["mtg_comprehensive_rules_index"][-1]["rules"]
    assert "Use this Magic Comprehensive Rules index" in instructions
    assert "601. Casting Spells" in instructions


def test_request_constraints_payload_exposes_must_include_and_avoid_terms() -> None:
    request = DeckRequest(
        format=Format.modern,
        must_include=["Slickshot Show-Off"],
        avoid=["Ragavan, Nimble Pilferer", "fetch lands"],
    )

    payload = request_constraints_payload(request)

    assert payload["must_include"] == ["Slickshot Show-Off"]
    assert payload["avoid"] == ["Ragavan, Nimble Pilferer", "fetch lands"]
    assert "Must include these cards" in payload["instructions"]
    assert "Avoid these card names" in payload["instructions"]


def test_openai_rag_planner_receives_workflow_skill(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_structured_openai_agent(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "thoughts": [],
            "strategy_queries": ["modern izzet prowess"],
            "meta_deck_queries": [],
            "rules_queries": ["modern deck construction"],
        }

    monkeypatch.setattr(
        "app.agent.deck_builder.run_structured_openai_agent",
        fake_run_structured_openai_agent,
    )
    agent = OpenAIDeckAgent(
        settings=Settings(OPENAI_API_KEY="test-key"),
        tools=FakeTools(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        agent._plan_rag(
            request=DeckRequest(
                format=Format.modern,
                colors=["U", "R"],
                playstyle="tempo",
                strategy="prowess",
                must_include=["Slickshot Show-Off"],
                avoid=["Ragavan, Nimble Pilferer"],
            ),
            rules_context=[],
            strategy_context=[],
        )
    )

    assert result["strategy_queries"] == ["modern izzet prowess"]
    assert "deck_builder_workflow skill" in captured["instructions"]
    assert "Slickshot Show-Off" in captured["instructions"]
    assert captured["input_payload"]["deck_workflow"]["current_phase"] == "rag_planning"
    assert captured["input_payload"]["request_constraints"]["must_include"] == ["Slickshot Show-Off"]
    assert captured["input_payload"]["request_constraints"]["avoid"] == ["Ragavan, Nimble Pilferer"]
    assert "search_cards_scryfall" not in captured["input_payload"]["deck_workflow"]["phase_allowed_tools"]


def test_agent_scryfall_phase_looks_up_must_include_cards() -> None:
    agent = OpenAIDeckAgent(
        settings=Settings(OPENAI_API_KEY="test-key"),
        tools=FakeTools(),  # type: ignore[arg-type]
    )
    steps: list[dict[str, str]] = []

    results = asyncio.run(
        agent._run_scryfall_tool_plan(
            request=DeckRequest(format=Format.modern, must_include=["Slickshot Show-Off"]),
            plan={"scryfall_queries": []},
            rag_context={"strategy": [], "meta_decks": [], "rules": []},
            steps=steps,
            rag_results={"strategy": [], "meta_decks": [], "rules": []},
        )
    )

    assert [payload["title"] for payload in results["lookups"]] == ["Slickshot Show-Off"]
    assert steps[0] == {
        "label": "Requested card lookup",
        "detail": "Slickshot Show-Off -> live Scryfall candidate",
    }
