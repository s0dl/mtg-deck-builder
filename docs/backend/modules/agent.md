# Agent Module

Path: `app/agent`

## Purpose

The agent module coordinates model-controlled deck construction without giving the model direct access to arbitrary code. The model can plan RAG searches; the backend executes only the allowed tools.

## Main Files

- `deck_builder.py` - OpenAI and Ollama agent implementations.
- `tools.py` - constrained RAG, Scryfall, and validation tools.

## OpenAI Agent Flow

1. Receive the user request plus initial RAG strategy/rules context.
2. Ask the model for extra RAG strategy/meta/rules queries.
3. Execute those RAG tools and collect the returned documents.
4. Ask the model for live Scryfall searches using the retrieved documents.
5. Execute the live Scryfall searches.
6. Send tool results back for card selection.
7. Return selected cards, explanation, live card payloads, and agent steps.

The OpenAI agent intentionally receives rules and strategy context first. It then uses the retrieved RAG documents to drive guarded live Scryfall searches for candidate cards. Live Scryfall is also called by the backend after deterministic validation to refresh prices and card facts.

## Tools

- `search_strategy(query, mtg_format, limit)` searches `mtgdecks_articles`.
- `search_rules(intent, limit)` searches `mtg_comprehensive_rules`.
- `search_cards_scryfall(query, limit)` searches live Scryfall for candidate cards.
- `lookup_card(name)` remains available for backend-controlled live checks.
- `validate_deck_cards(cards, mtg_format)` runs deterministic validation.

Tool responses are normalized as retrieved-document payloads so they can be displayed and reused by the API layer.
