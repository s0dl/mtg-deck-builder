# MCP Tool Layer

This directory contains the request-scoped MCP tool server and corpus-backed Magic data adapters.

## Main Files

- `server.py` registers and executes deck-builder MCP tools for RAG, card corpus search, and validation.
- `external.py` exposes the same tool surface through the MCP Python SDK for stdio or streamable HTTP transport.
- `app/rag/card_documents.py` contains the card response-to-document mapper used by ingestion and corpus compatibility code.

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

- Keep RAG, corpus-backed card search, and validation calls behind one constrained tool registry.
- Keep the external MCP transport aligned with the in-process registry so agents and other clients see the same tools.
- Refresh changed card metadata on a schedule.
- Push changed card text or legality documents into the RAG ingestion pipeline when those changes affect retrieval.

## Boundary

Do not put strategic opinions here. This layer should return indexed facts. Strategy belongs in RAG and deterministic constraints belong in skills.
