# Live Data Adapter Layer

This directory currently contains live Magic data adapters. It is not an MCP protocol server yet: there is no MCP transport, server manifest, or externally callable MCP tool registry in this project.

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

## MCP Direction

To make this a real MCP integration, expose the Scryfall operations through an MCP server process and have the agent call those tools through the MCP protocol instead of importing `ScryfallClient` directly.
