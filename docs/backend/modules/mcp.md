# Live Data Adapter Module

Path: `app/mcp`

## Purpose

This module contains live-data clients. Scryfall is the current source of truth for live card lookup, legality, and prices.

Despite the directory name, this is not currently a protocol-native MCP server. It is an internal adapter that the backend agent tools call directly.

## Main File

- `scryfall_client.py` - async Scryfall search and named-card lookup.

## Responsibilities

- Query live Scryfall search when agent or fallback flows need live candidates.
- Look up specific card names after validation for price and fact refresh.
- Convert Scryfall card JSON into retrieved-document objects.
- Preserve legality, colors, color identity, mana value, type line, oracle text, and price metadata when available.

The OpenAI agent uses guarded live Scryfall searches for candidate discovery. The backend also uses live Scryfall after validation for current prices and card facts.
