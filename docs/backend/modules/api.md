# API Module

Path: `app/api`

## Purpose

The API module exposes HTTP routes and coordinates request-scoped work.

## Main Files

- `decks.py` - `POST /api/decks/generate`.
- `agent.py` - `GET /api/agent/status`.

## Deck Generation Responsibilities

`decks.py` performs the backend orchestration:

- Build retrieval queries from the request.
- Retrieve strategy, rule, and meta context.
- Run OpenAI or Ollama agent flows when configured.
- Fall back to deterministic assembly when model paths fail.
- Convert agent live Scryfall payloads into card context.
- Merge duplicate card rows.
- Enforce target deck size.
- Add basic lands.
- Run Scryfall price enrichment after deterministic validation.
- Validate the final deck.
- Return agent steps and retrieved context for the frontend.

## Status Responsibilities

`agent.py` reports whether OpenAI or Ollama is configured and, for Ollama, whether the local server is reachable.
