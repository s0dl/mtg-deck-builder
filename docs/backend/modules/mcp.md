# MCP Tool Module

Path: `app/mcp`

## Purpose

This module owns the backend's constrained MCP tool server. It is the only layer that should execute agent-visible RAG tools, live Scryfall tools, card corpus lookup, or deck validation tools.

## Main File

- `server.py` - request-scoped MCP tool registry and executor.
- `external.py` - FastMCP transport wrapper for running the same tool surface outside the FastAPI request path.
- `scryfall_client.py` - async Scryfall search and named-card lookup.

## Responsibilities

- Expose tool schemas for RAG text/vector searches, rule-prefix lookups, strategy/meta searches, Scryfall live search, exact card lookup, card corpus search, and validation.
- Route backend and agent tool calls through `DeckBuilderMcpServer.call_tool()` or `call_tool_sync()`.
- Expose the same tool set through the MCP Python SDK for external stdio or streamable HTTP clients.
- Query live Scryfall search when agent or fallback flows need live candidates.
- Look up specific card names after validation for price and fact refresh.
- Convert Scryfall card JSON into retrieved-document objects.
- Preserve legality, colors, color identity, mana value, type line, oracle text, and price metadata when available.

The OpenAI and Ollama agents receive tool signatures from this server. Initial RAG context, fallback live candidate discovery, and post-validation price refresh also route through the same MCP server boundary.

## Running A Local MCP Server

`app/main.py` does not expose MCP transport routes. It only serves the FastAPI API under `/api/*` and `/health`. If you want an external MCP client such as Codex, Claude Code, or another MCP-aware tool to connect directly, run `app.mcp.external` as a separate process.

### Prerequisites

- A working Postgres database with the deck-builder schema available.
- `DATABASE_URL` pointing at that database.
- The backend dependencies installed in the environment you will launch the server from.

### Stdio Transport

Use stdio when the client launches the MCP server as a subprocess and speaks MCP over stdin/stdout.

```bash
cd backend
source .venv/bin/activate
DATABASE_URL=postgresql+psycopg://mtg:mtg@localhost:5432/mtg_deck_builder \
MCP_TRANSPORT=stdio \
python -m app.mcp.external
```

Client-side configuration should point at the command above, not at the FastAPI app. The important parts are:

- command: `python -m app.mcp.external`
- env: `DATABASE_URL=...`
- env: `MCP_TRANSPORT=stdio`

### Streamable HTTP Transport

Use streamable HTTP when the client wants to connect to a local HTTP endpoint.

```bash
cd backend
source .venv/bin/activate
DATABASE_URL=postgresql+psycopg://mtg:mtg@localhost:5432/mtg_deck_builder \
MCP_TRANSPORT=streamable-http \
MCP_HOST=127.0.0.1 \
MCP_PORT=8010 \
python -m app.mcp.external
```

Point the client at:

```text
http://127.0.0.1:8010/mcp
```

If you see `404 Not Found` for `/mcp`, the usual cause is that the client is talking to the FastAPI service instead of the separate MCP process, or the MCP process is not running with `MCP_TRANSPORT=streamable-http`.

### Practical Client Setup

For a client that supports MCP subprocesses, use the stdio command. For a client that supports streamable HTTP, use the URL above. The exact config syntax varies by product, but the key values are always the same:

- the server command is `python -m app.mcp.external`
- the working directory is `backend`
- `DATABASE_URL` must be set
- `MCP_TRANSPORT` selects `stdio` or `streamable-http`
