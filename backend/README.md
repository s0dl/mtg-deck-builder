# Backend

Python FastAPI service for MTG Deck Builder Agent.

## Responsibilities

- Expose the deck generation API.
- Orchestrate MCP, RAG, and deterministic skills.
- Store and query pgvector knowledge documents.
- Provide ingestion script entry points.

## Layout

- `app/api` - HTTP route modules.
- `app/core` - configuration and shared infrastructure.
- `app/mcp` - live-data clients, starting with Scryfall.
- `app/rag` - pgvector document models, repository, and retrieval.
- `app/skills` - deterministic deck validation and scoring functions.
- `app/models` - request/response schemas.
- `scripts` - ingestion and maintenance jobs.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

## Tests

```bash
pytest
```

## RAG Commands

Seed a small local corpus:

```bash
python -m scripts.seed_rag
```

Download and import Scryfall default cards:

```bash
python -m scripts.download_scryfall_bulk --output data/scryfall-default-cards.json
python -m scripts.ingest_scryfall_bulk data/scryfall-default-cards.json
```
