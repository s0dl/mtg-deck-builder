# Agent Module

Path: `app/agent`

## Purpose

The agent module coordinates model-controlled deck construction without giving the model direct access to arbitrary code. The model can plan RAG searches; the backend executes only the allowed tools.

## Main Files

- `deck_builder.py` - OpenAI and Ollama agent implementations.
- `tools.py` - constrained RAG, Scryfall, and validation tools.

## OpenAI Agent Flow

1. Receive the user request plus initial RAG strategy/rules context.
2. Ask the model for extra RAG strategy/rules queries and live Scryfall card queries.
3. Execute allowed backend tools.
4. Send tool results back for card selection.
5. Return selected cards, explanation, live card payloads, and agent steps.

The OpenAI agent intentionally receives rules and strategy context first. It then uses guarded live Scryfall searches for candidate cards. Live Scryfall is also called by the backend after deterministic validation to refresh prices and card facts.

## Tools

- `search_strategy(query, mtg_format, limit)` searches `mtgdecks_articles`.
- `search_rules(intent, limit)` searches `mtg_comprehensive_rules`.
- `search_cards_scryfall(query, limit)` searches live Scryfall for candidate cards.
- `lookup_card(name)` remains available for backend-controlled live checks.
- `validate_deck_cards(cards, mtg_format)` runs deterministic validation.

Tool responses are normalized as retrieved-document payloads so they can be displayed and reused by the API layer.
