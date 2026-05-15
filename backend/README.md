# Backend

Python FastAPI service for MTG Deck Builder Agent.

## Responsibilities

- Expose deck generation and agent status APIs.
- Orchestrate OpenAI Agents SDK/Ollama agent flows, MCP-routed RAG retrieval, live Scryfall tools, and deterministic skills.
- Store and query pgvector knowledge documents.
- Download and ingest rules, articles, meta deck snapshots, and Scryfall card corpus records.

## Documentation

- [Backend Docs Index](../docs/backend/README.md)
- [API Reference](../docs/backend/apis.md)
- [Agent Module](../docs/backend/modules/agent.md)
- [API Module](../docs/backend/modules/api.md)
- [Core Module](../docs/backend/modules/core.md)
- [LLM Module](../docs/backend/modules/llm.md)
- [MCP Module](../docs/backend/modules/mcp.md)
- [Models Module](../docs/backend/modules/models.md)
- [RAG Module](../docs/backend/modules/rag.md)
- [Scripts Module](../docs/backend/modules/scripts.md)
- [Skills Module](../docs/backend/modules/skills.md)

Module-local implementation notes are also kept under `app/*/README.md` where useful.

## Development

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,agent]"
uvicorn app.main:app --reload
```

## Tests

```bash
pytest
python -m ruff check app tests scripts
```

## Common Operations

Check agent configuration:

```bash
curl http://localhost:8000/api/agent/status
```

Run ingestion from the same environment that can reach Postgres. In Docker, use `docker compose exec backend ...`. From the host, make sure `DATABASE_URL` points at `localhost`, not Docker's internal `postgres` hostname.
