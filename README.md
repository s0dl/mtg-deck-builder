# MTG Deck Builder Agent

AI-assisted Magic: The Gathering deck building app. The backend combines RAG context, live Scryfall card lookups, model-driven agent planning, and deterministic deck validation. The frontend exposes the generation workflow and shows retrieved context, agent activity, validation, pricing, and the final deck list.

## Project Layout

- `backend/` - FastAPI service, agent orchestration, RAG, live Scryfall tools, deterministic skills, and ingestion scripts.
- `frontend/` - React + TypeScript UI for deck requests and generation results.
- `database/` - Postgres/pgvector initialization.
- `docs/` - Cross-cutting architecture and handoff notes.

## Documentation

- [Docs Index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Current Handoff](docs/handoff.md)
- [Backend Docs](docs/backend/README.md)
- [Backend API Reference](docs/backend/apis.md)
- [Frontend Docs](docs/frontend/README.md)

All project docs live under `docs/`.

## Quick Start

```bash
cp .env.example .env
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
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,agent]"
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Current Generation Flow

The preferred path is the OpenAI agent flow. RAG provides initial rules and strategy context, then the model plans extra strategy/rules searches and live Scryfall searches for card candidates. The backend validates and finalizes the deck, then calls Scryfall for current prices.

If model generation fails or is disabled, the backend falls back to deterministic assembly.
