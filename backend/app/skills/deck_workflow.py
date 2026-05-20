from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.models.deck import DeckRequest

WorkflowPhase = Literal["rag_planning", "scryfall_planning", "card_selection", "validation"]


@dataclass(frozen=True)
class DeckWorkflowStep:
    key: str
    objective: str
    tools: tuple[str, ...]
    output: str


MTG_COMPREHENSIVE_RULES_INDEX: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "1. Game Concepts",
        (
            "100. General",
            "101. The Magic Golden Rules",
            "102. Players",
            "103. Starting the Game",
            "104. Ending the Game",
            "105. Colors",
            "106. Mana",
            "107. Numbers and Symbols",
            "108. Cards",
            "109. Objects",
            "110. Permanents",
            "111. Tokens",
            "112. Spells",
            "113. Abilities",
            "114. Emblems",
            "115. Targets",
            "116. Special Actions",
            "117. Timing and Priority",
            "118. Costs",
            "119. Life",
            "120. Damage",
            "121. Drawing a Card",
            "122. Counters",
            "123. Stickers",
        ),
    ),
    (
        "2. Parts of a Card",
        (
            "200. General",
            "201. Name",
            "202. Mana Cost and Color",
            "203. Illustration",
            "204. Color Indicator",
            "205. Type Line",
            "206. Expansion Symbol",
            "207. Text Box",
            "208. Power/Toughness",
            "209. Loyalty",
            "210. Defense",
            "211. Hand Modifier",
            "212. Life Modifier",
            "213. Information Below the Text Box",
        ),
    ),
    (
        "3. Card Types",
        (
            "300. General",
            "301. Artifacts",
            "302. Creatures",
            "303. Enchantments",
            "304. Instants",
            "305. Lands",
            "306. Planeswalkers",
            "307. Sorceries",
            "308. Kindreds",
            "309. Dungeons",
            "310. Battles",
            "311. Planes",
            "312. Phenomena",
            "313. Vanguards",
            "314. Schemes",
            "315. Conspiracies",
        ),
    ),
    (
        "4. Zones",
        (
            "400. General",
            "401. Library",
            "402. Hand",
            "403. Battlefield",
            "404. Graveyard",
            "405. Stack",
            "406. Exile",
            "407. Ante",
            "408. Command",
        ),
    ),
    (
        "5. Turn Structure",
        (
            "500. General",
            "501. Beginning Phase",
            "502. Untap Step",
            "503. Upkeep Step",
            "504. Draw Step",
            "505. Main Phase",
            "506. Combat Phase",
            "507. Beginning of Combat Step",
            "508. Declare Attackers Step",
            "509. Declare Blockers Step",
            "510. Combat Damage Step",
            "511. End of Combat Step",
            "512. Ending Phase",
            "513. End Step",
            "514. Cleanup Step",
        ),
    ),
    (
        "6. Spells, Abilities, and Effects",
        (
            "600. General",
            "601. Casting Spells",
            "602. Activating Activated Abilities",
            "603. Handling Triggered Abilities",
            "604. Handling Static Abilities",
            "605. Mana Abilities",
            "606. Loyalty Abilities",
            "607. Linked Abilities",
            "608. Resolving Spells and Abilities",
            "609. Effects",
            "610. One-Shot Effects",
            "611. Continuous Effects",
            "612. Text-Changing Effects",
            "613. Interaction of Continuous Effects",
            "614. Replacement Effects",
            "615. Prevention Effects",
            "616. Interaction of Replacement and/or Prevention Effects",
        ),
    ),
    (
        "7. Additional Rules",
        (
            "700. General",
            "701. Keyword Actions",
            "702. Keyword Abilities",
            "703. Turn-Based Actions",
            "704. State-Based Actions",
            "705. Flipping a Coin",
            "706. Rolling a Die",
            "707. Copying Objects",
            "708. Face-Down Spells and Permanents",
            "709. Split Cards",
            "710. Flip Cards",
            "711. Leveler Cards",
            "712. Double-Faced Cards",
            "713. Substitute Cards",
            "714. Saga Cards",
            "715. Adventurer Cards",
            "716. Class Cards",
            "717. Attraction Cards",
            "718. Prototype Cards",
            "719. Case Cards",
            "720. Omen Cards",
            "721. Station Cards",
            "722. Preparation Cards",
            "723. Controlling Another Player",
            "724. Ending Turns and Phases",
            "725. The Monarch",
            "726. The Initiative",
            "727. Restarting the Game",
            "728. Rad Counters",
            "729. Subgames",
            "730. Merging with Permanents",
            "731. Day and Night",
            "732. Taking Shortcuts",
            "733. Handling Illegal Actions",
        ),
    ),
    (
        "8. Multiplayer Rules",
        (
            "800. General",
            "801. Limited Range of Influence Option",
            "802. Attack Multiple Players Option",
            "803. Attack Left and Attack Right Options",
            "804. Deploy Creatures Option",
            "805. Shared Team Turns Option",
            "806. Free-for-All Variant",
            "807. Grand Melee Variant",
            "808. Team vs. Team Variant",
            "809. Emperor Variant",
            "810. Two-Headed Giant Variant",
            "811. Alternating Teams Variant",
        ),
    ),
    (
        "9. Casual Variants",
        (
            "900. General",
            "901. Planechase",
            "902. Vanguard",
            "903. Commander",
            "904. Archenemy",
            "905. Conspiracy Draft",
        ),
    ),
)


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
    payload: dict[str, Any] = {
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
    if current_phase == "rag_planning":
        payload["mtg_comprehensive_rules_index"] = comprehensive_rules_index_payload()
    return payload


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
    if current_phase == "rag_planning":
        lines.append("Use this Magic Comprehensive Rules index to target search_rules queries:")
        lines.append(comprehensive_rules_index_text())
    lines.append("Hard rules:")
    lines.extend(f"- {rule}" for rule in HARD_RULES)
    return "\n".join(lines)


def comprehensive_rules_index_payload() -> list[dict[str, Any]]:
    """Return the top-level Comprehensive Rules index for model query planning."""
    return [
        {"section": section, "rules": list(rules)}
        for section, rules in MTG_COMPREHENSIVE_RULES_INDEX
    ]


def comprehensive_rules_index_text() -> str:
    """Return a compact prompt-readable Comprehensive Rules index."""
    lines: list[str] = []
    for section, rules in MTG_COMPREHENSIVE_RULES_INDEX:
        lines.append(section)
        lines.extend(rules)
    return "\n".join(lines)


def request_constraints_payload(request: DeckRequest) -> dict[str, Any]:
    """Return explicit model-facing request constraints beyond the raw request JSON."""
    return {
        "must_include": list(request.must_include),
        "avoid": list(request.avoid),
        "instructions": request_constraints_instructions(request),
    }


def request_constraints_instructions(request: DeckRequest) -> str:
    """Return prompt text for must-include and avoid constraints."""
    lines: list[str] = []
    if request.must_include:
        lines.append(
            "Must include these cards if they are legal, in color identity, and available in candidates: "
            + ", ".join(request.must_include)
            + "."
        )
    else:
        lines.append("No explicit must-include cards were requested.")

    if request.avoid:
        lines.append(
            "Avoid these card names, mechanics, or terms when planning searches and selecting cards: "
            + ", ".join(request.avoid)
            + "."
        )
    else:
        lines.append("No explicit avoid terms were requested.")

    return " ".join(lines)
