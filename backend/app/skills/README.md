# Skill Functions

Skill functions are deterministic checks and calculations. They should be pure or close to pure so they are easy to unit test.

## Current Modules

- `deck_rules.py` - deck size and copy-limit validation.
- `mana_curve.py` - mana curve aggregation.
- `filters.py` - budget, color, and type filters.
- `synergy.py` - tag-based synergy helpers.
- `deck_evaluation.py` - candidate legality, color identity, curve, role, budget, and synergy scoring.
- `deck_workflow.py` - ordered agent workflow contract for RAG, live Scryfall, card selection, and validation.

## Production Direction

These functions should be called after candidate generation and before returning a deck. The agent can explain tradeoffs, but the skill layer should be the source of truth for objective rule failures.

The workflow skill is fed into each model-planning phase so the agent uses MCP tools in a consistent order: RAG first, live Scryfall second, model card selection third, deterministic validation last.

Next, feed evaluation summaries back into user-visible agent steps and final deck explanations.
