# Skills Module

Path: `app/skills`

## Purpose

Deterministic helper functions for constraints and scoring signals that should not rely on model judgment.

## Files

- `deck_rules.py` - deck size and copy-limit validation.
- `mana_curve.py` - mana curve aggregation.
- `filters.py` - budget, color, and type filtering helpers.
- `synergy.py` - tag-based synergy helpers.

## Direction

The next major improvement should be a deck evaluation/building skill that scores candidates against:

- Format legality.
- Color identity.
- Deck size and copy limits.
- Mana curve.
- Role balance.
- Budget.
- Synergy with requested strategy and selected cards.
- Meta deck context.

The agent can propose, but skills should decide whether a result satisfies objective constraints.
