# Skills Module

Path: `app/skills`

## Purpose

Deterministic helper functions for constraints and scoring signals that should not rely on model judgment.

## Files

- `deck_rules.py` - deck size and copy-limit validation.
- `mana_curve.py` - mana curve aggregation.
- `filters.py` - budget, color, and type filtering helpers.
- `synergy.py` - tag-based synergy helpers.
- `deck_evaluation.py` - candidate legality, color identity, curve, role, budget, and synergy scoring.
- `deck_workflow.py` - ordered workflow contract used by agent prompts and payloads.

## Workflow Skill

`deck_workflow.py` defines the deck-building sequence the agent must follow:

1. Read request constraints.
2. Retrieve strategy, meta deck, and rules context through RAG MCP tools.
3. Discover current cards through live Scryfall MCP tools.
4. Select exact candidate names with counts and roles.
5. Let deterministic backend skills validate, finalize, and refresh prices.

Each agent phase receives the workflow payload plus phase-specific allowed tools. This keeps the model from planning live card searches before RAG retrieval or selecting cards before live candidates exist.

## Direction

The next major improvement should refine the deck evaluation/building skill so it can enforce:

- Format legality.
- Color identity.
- Deck size and copy limits.
- Mana curve.
- Role balance.
- Budget.
- Synergy with requested strategy and selected cards.
- Meta deck context.

The agent can propose, but skills should decide whether a result satisfies objective constraints.
