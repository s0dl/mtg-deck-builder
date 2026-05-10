# RAG Layer

The RAG layer provides long-lived card, deck, meta, and strategy knowledge using Postgres with pgvector.

## Sources

- Scryfall bulk card dump for complete card text and mechanics.
- Recent tournament results and decklists, weighted toward the last 1-2 years.
- Format primers from trusted strategy sources.
- Foundational articles for evergreen heuristics.

## Storage

The initial schema is in `database/init/001_pgvector.sql`.

`rag_documents` stores:

- `source` - provider name, such as `scryfall_bulk` or `mtggoldfish`.
- `source_id` - stable upstream identifier.
- `title` - human-readable label.
- `content` - chunk text.
- `metadata` - format, archetype, colors, date, card names, and other filters.
- `embedding` - pgvector embedding.

## Retrieval

`retriever.py` uses pgvector cosine similarity when embeddings are present and falls back to text search when the table is empty.

The default `hash` embedding provider is deterministic and local. It is meant for development, smoke tests, and proving the pipeline. It is not a semantic embedding model. Production should add a provider in `embeddings.py`, set `EMBEDDING_PROVIDER`, and keep the repository/retriever contracts unchanged.

## Commands

Seed local strategy documents:

```bash
python -m scripts.seed_rag
```

Download and import Scryfall default cards:

```bash
python -m scripts.download_scryfall_bulk --output data/scryfall-default-cards.json
python -m scripts.ingest_scryfall_bulk data/scryfall-default-cards.json
```
