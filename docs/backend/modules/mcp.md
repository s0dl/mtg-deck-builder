# MCP Module

Path: `app/mcp`

## Purpose

The MCP module contains live-data clients. Scryfall is the current source of truth for live card lookup, legality, and prices.

## Main File

- `scryfall_client.py` - async Scryfall search and named-card lookup.

## Responsibilities

- Query live Scryfall search when fallback flows need live candidates.
- Look up specific card names after validation for price and fact refresh.
- Convert Scryfall card JSON into retrieved-document objects.
- Preserve legality, colors, color identity, mana value, type line, oracle text, and price metadata when available.

The OpenAI agent should use RAG card corpus records for candidate discovery. The backend should use live Scryfall after validation for current prices and card facts. Fallback flows may still use live Scryfall to assemble candidates when the agent path is unavailable.
