# MCP Layer

The MCP layer owns live Magic data that can change frequently and should not be trusted from stale embeddings.

## Initial Provider

`scryfall_client.py` wraps Scryfall API calls for:

- Card lookup
- Search
- Price fields
- Legality fields

## Production Responsibilities

- Fetch current card data at request time when price or legality matters.
- Refresh changed card metadata on a schedule.
- Push changed card text or legality documents into the RAG ingestion pipeline when those changes affect retrieval.

## Boundary

Do not put strategic opinions here. This layer should return live facts. Strategy belongs in RAG and deterministic constraints belong in skills.
