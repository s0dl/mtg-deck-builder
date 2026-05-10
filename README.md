# MTG Deck Builder Agent

An AI-powered Magic: The Gathering deck building assistant that combines live MCP data, pgvector-backed RAG, and deterministic skill functions. A user describes the format, budget, playstyle, and strategy they want, and the agent returns a coherent deck with explanations.

## Architecture

This repo is organized as a small monorepo:

- `backend/` - Python FastAPI service that owns agent orchestration, MCP clients, RAG retrieval, and deterministic deck-building skills.
- `frontend/` - TypeScript React app for entering deck goals and viewing generated deck plans.
- `database/` - Postgres/pgvector initialization scripts.
- `docs/` - Cross-cutting architecture notes.

## Layers

### MCP Layer

Live data belongs in `backend/app/mcp`.

This layer is responsible for frequently changing truth:

- Scryfall card lookups
- Prices
- Format legality
- Ban list changes

See `backend/app/mcp/README.md`.

### RAG Layer

The retrieval layer belongs in `backend/app/rag` and uses Postgres with `pgvector`.

Initial production sources:

- Scryfall bulk card dump for card text, mechanics, and keywords
- Recent tournament results and decklists from the last 1-2 years
- Format primers and strategy articles from MTGGoldfish and ChannelFireball
- Foundational strategy content for timeless deck-building principles

See `backend/app/rag/README.md`.

### Skill Functions

Deterministic tools belong in `backend/app/skills`.

These functions validate and score objective deck constraints:

- Deck construction rules
- Four-copy limit
- Format legality
- Mana curve
- Budget filtering
- Color identity and type filtering
- Synergy constraints

See `backend/app/skills/README.md`.

## Ingestion Pipeline

The production-ready path is:

1. Import Scryfall bulk JSON into normalized Postgres tables.
2. Chunk card, decklist, and article knowledge into embedding documents.
3. Store embeddings in pgvector.
4. Schedule MCP refresh jobs for price, legality, and ban-list deltas.
5. Re-embed only changed knowledge documents.

The current scaffold includes the storage schema and script entry points, but does not download or embed external data by default.

## Quick Start

Copy environment defaults:

```bash
cp .env.example .env
```

Bring up the stack:

```bash
docker compose up --build
```

Services:

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs
- Postgres: localhost:5432

## Local Development

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Current Status

This is a scaffold with working service boundaries, schemas, Docker setup, and placeholder agent behavior. The next implementation steps are:

- Add a real embedding provider.
- Implement Scryfall bulk import in `backend/scripts/ingest_scryfall_bulk.py`.
- Add scheduled MCP refresh jobs.
- Replace the placeholder deck generation with retrieval-augmented ranking and construction logic.
