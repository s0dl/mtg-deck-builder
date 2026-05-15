# Core Module

Path: `app/core`

## Purpose

Shared backend infrastructure.

## Files

- `config.py` - environment-backed settings.
- `database.py` - SQLAlchemy engine/session setup.
- `logging.py` - request-aware logging helpers.

## Important Settings

- `DATABASE_URL`
- `BACKEND_CORS_ORIGINS`
- `SCRYFALL_API_BASE_URL`
- `AGENT_PROVIDER`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `EMBEDDING_PROVIDER`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSIONS`
- `EMBEDDING_CACHE_ENABLED`
- `RAG_INGEST_BATCH_SIZE`
- `RAG_INGEST_BATCH_DELAY_SECONDS`

When running scripts on the host, use a host-reachable database URL such as `localhost`. When running inside Docker Compose, use the service hostname `postgres`.
