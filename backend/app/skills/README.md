# Skill Functions

Skill functions are deterministic checks and calculations. They should be pure or close to pure so they are easy to unit test.

## Current Modules

- `deck_rules.py` - deck size and copy-limit validation.
- `mana_curve.py` - mana curve aggregation.
- `filters.py` - budget, color, and type filters.
- `synergy.py` - synergy constraint scaffolding.

## Production Direction

These functions should be called after candidate generation and before returning a deck. The agent can explain tradeoffs, but the skill layer should be the source of truth for objective rule failures.
