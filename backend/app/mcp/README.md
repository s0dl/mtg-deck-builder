# MCP Tool Layer

This directory contains the request-scoped MCP tool server and live Magic data adapters.

## Main Files

- `server.py` registers and executes deck-builder MCP tools for RAG, live Scryfall, card corpus search, and validation.
- `scryfall_client.py` wraps Scryfall API calls for card lookup, search, price fields, and legality fields.

## Tools

`DeckBuilderMcpServer` exposes:

- `search_rag_text`
- `search_rag_vector`
- `search_rag_metadata_prefixes`
- `search_strategy`
- `search_meta_decks`
- `search_rules`
- `search_card_corpus`
- `search_cards_scryfall`
- `lookup_card`
- `validate_deck_cards`

## Production Responsibilities

- Keep RAG, Scryfall, and validation calls behind one constrained tool registry.
- Fetch current card data at request time when price or legality matters.
- Refresh changed card metadata on a schedule.
- Push changed card text or legality documents into the RAG ingestion pipeline when those changes affect retrieval.

## Boundary

Do not put strategic opinions here. This layer should return live facts. Strategy belongs in RAG and deterministic constraints belong in skills.
