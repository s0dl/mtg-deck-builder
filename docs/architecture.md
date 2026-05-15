# Architecture

MTG Deck Builder is a FastAPI + React application that builds Magic: The Gathering deck drafts from three categories of knowledge:

- Live card truth from Scryfall.
- Indexed rules, strategy, meta, and card text context from Postgres/pgvector RAG.
- Deterministic validation and shaping skills for constraints that should not depend on a model.

## Runtime Components

### Frontend

The React frontend owns the user workflow. It captures deck goals, calls `POST /api/decks/generate`, then renders the deck list, validation, mana curve, price estimate, retrieved context, and agent activity timeline.

The frontend does not construct or validate decks.

### Backend API

The FastAPI backend exposes:

- `GET /health`
- `GET /api/agent/status`
- `POST /api/decks/generate`

The backend assembles request context, runs the configured agent path, calls constrained tools, validates the deck, and returns a typed response.

### Agent Layer

The preferred generation path is the OpenAI agent in `backend/app/agent`.

RAG supplies rules and strategy context first. The model then decides which constrained tools to call:

- `search_strategy`
- `search_rules`
- `search_cards_scryfall`
- `validate_deck_cards`

Tool calls are executed by the backend, not by arbitrary model-side code. Live Scryfall payloads become candidate card context and are reused after deterministic validation to refresh prices and card facts.

Ollama remains available as an optional local experiment, but smaller local models may fail to produce stable structured deck output.

### RAG Layer

Postgres stores `rag_documents` with pgvector embeddings. Current sources include:

- `mtg_comprehensive_rules`
- `mtgdecks_articles`
- `mtgdecks_meta_decks`
- `scryfall_bulk`

The current OpenAI agent flow treats RAG as rules, strategy, meta, and semantic card context. Live Scryfall remains the source of truth for current prices and card facts after validation.

### Deterministic Skills

The skill layer handles objective checks and summary calculations:

- Deck size and copy limits.
- Commander singleton behavior.
- Mana curve.
- Basic color/type/budget helpers.
- Synergy tag helpers.

The backend also finalizes model-selected cards by merging duplicates, trimming over-sized results, adding basic lands, validating the final response, and then enriching prices from Scryfall.

## Deck Generation Flow

1. The frontend submits a `DeckRequest`.
2. FastAPI creates a request-scoped database session.
3. The backend retrieves strategy, rules, and meta context from RAG.
4. If the OpenAI agent is enabled, the model receives that context and plans extra RAG calls for strategy/rules plus live Scryfall card searches.
5. The backend executes those constrained tool calls.
6. The model selects cards from the returned live Scryfall context.
7. The backend merges, trims, sizes, and validates the deck.
8. The backend calls Scryfall to refresh prices after validation.
9. If the agent path fails, the backend falls back to deterministic construction.
10. The response includes cards, validation, mana curve, retrieved context, `agent_steps`, and `generation_mode`.

## Data Flow

```text
React UI
  -> FastAPI /api/decks/generate
  -> RAG retrieval for rules/strategy/meta
  -> agent planning
  -> backend RAG tool execution
  -> model card selection
  -> deterministic finalization
  -> Scryfall price checks
  -> typed DeckResponse
```

## Configuration Modes

OpenAI agent and OpenAI embeddings:

```bash
AGENT_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=384
```

Local embeddings with model generation disabled or fallback-only:

```bash
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSIONS=384
```

Smoke-test embeddings:

```bash
EMBEDDING_PROVIDER=hash
```

## Persistence

Docker Compose uses named volumes for Postgres and backend data. If embedding dimensions change, recreate the Postgres volume and re-ingest documents. Vectors cannot be safely reused across dimensions.

The embedding cache is versioned by provider/model settings. If document formatting changes, old rows in `rag_documents` should be upserted by re-ingestion; the cache only avoids repeating identical embedding requests.
