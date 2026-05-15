import asyncio
from typing import Any

from app.agent.deck_builder import OpenAIDeckAgent
from app.core.config import Settings
from app.models.deck import DeckRequest, Format
from app.skills.deck_workflow import (
    PHASE_ALLOWED_TOOLS,
    workflow_instructions,
    workflow_payload,
    workflow_tool_order,
)


class FakeTools:
    def tool_signatures(self, names: tuple[str, ...] | list[str]) -> list[str]:
        return [f"{name}()" for name in names]


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
            ),
            rules_context=[],
            strategy_context=[],
        )
    )

    assert result["strategy_queries"] == ["modern izzet prowess"]
    assert "deck_builder_workflow skill" in captured["instructions"]
    assert captured["input_payload"]["deck_workflow"]["current_phase"] == "rag_planning"
    assert "search_cards_scryfall" not in captured["input_payload"]["deck_workflow"]["phase_allowed_tools"]
