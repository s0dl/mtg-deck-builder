from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

WorkflowPhase = Literal["rag_planning", "scryfall_planning", "card_selection", "validation"]


@dataclass(frozen=True)
class DeckWorkflowStep:
    key: str
    objective: str
    tools: tuple[str, ...]
    output: str


WORKFLOW_STEPS: tuple[DeckWorkflowStep, ...] = (
    DeckWorkflowStep(
        key="read_request",
        objective=(
            "Identify format, colors, budget, playstyle, strategy, must-include cards, "
            "and avoid terms before planning any tool calls."
        ),
        tools=(),
        output="A concise constraint summary used by every later phase.",
    ),
    DeckWorkflowStep(
        key="retrieve_strategy_rules",
        objective=(
            "Use RAG before card discovery. Retrieve format rules, strategy articles, "
            "and meta deck snapshots that explain how this deck should be built."
        ),
        tools=("search_strategy", "search_meta_decks", "search_rules"),
        output="Rules, strategy, and meta context for live-card search planning.",
    ),
    DeckWorkflowStep(
        key="discover_live_cards",
        objective=(
            "Use retrieved context to build targeted live Scryfall searches for core cards, "
            "support pieces, interaction, engines, and mana fixing."
        ),
        tools=("search_cards_scryfall",),
        output="Live Scryfall candidate card documents with current legality, price, and text.",
    ),
    DeckWorkflowStep(
        key="select_candidates",
        objective=(
            "Select exact candidate names only. Balance roles, curve, budget, legality, "
            "color identity, and the requested strategy."
        ),
        tools=(),
        output="A focused card package with counts and short roles.",
    ),
    DeckWorkflowStep(
        key="validate_finalize",
        objective=(
            "Let deterministic backend skills merge duplicates, fill land slots, enforce "
            "copy limits, validate deck size, and refresh prices."
        ),
        tools=("validate_deck_cards", "lookup_card"),
        output="A validated DeckResponse.",
    ),
)

PHASE_ALLOWED_TOOLS: dict[WorkflowPhase, tuple[str, ...]] = {
    "rag_planning": ("search_strategy", "search_meta_decks", "search_rules"),
    "scryfall_planning": ("search_cards_scryfall",),
    "card_selection": (),
    "validation": ("validate_deck_cards", "lookup_card"),
}

PHASE_OBJECTIVES: dict[WorkflowPhase, str] = {
    "rag_planning": (
        "Plan only RAG searches. Do not plan Scryfall searches or select cards yet. "
        "Cover strategy, meta deck examples, and format-specific construction rules."
    ),
    "scryfall_planning": (
        "Plan only live Scryfall searches from retrieved RAG context. Use format, color "
        "identity, paper availability, and non-land or land filters explicitly."
    ),
    "card_selection": (
        "Choose exact names from candidates only. Prefer coherent packages over isolated "
        "staples, and leave final deck-size validation to deterministic backend skills."
    ),
    "validation": (
        "Run deterministic validation and price refresh after card selection. Do not rely "
        "on model judgment for objective deck legality."
    ),
}

HARD_RULES: tuple[str, ...] = (
    "Do not skip RAG retrieval before live card discovery.",
    "Do not invent card names; select from candidate payloads only.",
    "Do not ask for tools outside the current workflow phase.",
    "Treat budget_usd as a ceiling: optimize power within the ceiling, not cheapest-only output.",
    "The backend, not the model, is the source of truth for legality, copy limits, and final size.",
)


def workflow_tool_order() -> list[str]:
    """Return MCP tool names in the order the deck workflow expects them."""
    ordered: list[str] = []
    for step in WORKFLOW_STEPS:
        for tool in step.tools:
            if tool not in ordered:
                ordered.append(tool)
    return ordered


def workflow_payload(current_phase: WorkflowPhase) -> dict[str, Any]:
    """Return a compact, JSON-serializable workflow contract for model prompts."""
    return {
        "name": "deck_builder_workflow",
        "current_phase": current_phase,
        "phase_objective": PHASE_OBJECTIVES[current_phase],
        "phase_allowed_tools": list(PHASE_ALLOWED_TOOLS[current_phase]),
        "tool_order": workflow_tool_order(),
        "steps": [
            {
                "key": step.key,
                "objective": step.objective,
                "tools": list(step.tools),
                "output": step.output,
            }
            for step in WORKFLOW_STEPS
        ],
        "hard_rules": list(HARD_RULES),
    }


def workflow_instructions(current_phase: WorkflowPhase) -> str:
    """Return prompt instructions that bind a model call to one workflow phase."""
    lines = [
        "Follow the deck_builder_workflow skill.",
        f"Current phase: {current_phase}.",
        PHASE_OBJECTIVES[current_phase],
        "Ordered workflow:",
    ]
    for index, step in enumerate(WORKFLOW_STEPS, start=1):
        tool_text = ", ".join(step.tools) if step.tools else "no MCP tools"
        lines.append(f"{index}. {step.key}: {step.objective} Tools: {tool_text}.")
    lines.append("Hard rules:")
    lines.extend(f"- {rule}" for rule in HARD_RULES)
    return "\n".join(lines)
