# MCP Tool Module

Path: `app/mcp`

## Purpose

This module owns the backend's constrained MCP tool server. It is the only layer that should execute agent-visible RAG tools, live Scryfall tools, card corpus lookup, or deck validation tools.

## Main File

- `server.py` - request-scoped MCP tool registry and executor.
- `scryfall_client.py` - async Scryfall search and named-card lookup.

## Responsibilities

- Expose tool schemas for RAG text/vector searches, rule-prefix lookups, strategy/meta searches, Scryfall live search, exact card lookup, card corpus search, and validation.
- Route backend and agent tool calls through `DeckBuilderMcpServer.call_tool()` or `call_tool_sync()`.
- Query live Scryfall search when agent or fallback flows need live candidates.
- Look up specific card names after validation for price and fact refresh.
- Convert Scryfall card JSON into retrieved-document objects.
- Preserve legality, colors, color identity, mana value, type line, oracle text, and price metadata when available.

The OpenAI and Ollama agents receive tool signatures from this server. Initial RAG context, fallback live candidate discovery, and post-validation price refresh also route through the same MCP server boundary.
