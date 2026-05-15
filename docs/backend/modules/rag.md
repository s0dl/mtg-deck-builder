# RAG Module

Path: `app/rag`

## Purpose

The RAG module stores and retrieves indexed knowledge in Postgres with pgvector.

Runtime retrieval is executed through `app.mcp.server.DeckBuilderMcpServer`; this module owns storage and retrieval primitives, while the MCP layer owns tool schemas and access control.

## Main Files

- `documents.py` - SQLAlchemy document model.
- `repository.py` - upsert and search persistence.
- `retriever.py` - vector, text, and metadata retrieval facade.
- `embeddings.py` - OpenAI, FastEmbed, hash, and cache providers.
- `ingestion.py` - source-specific document builders.
- `chunking.py` - chunking helpers and Scryfall card text formatting.

## Sources

- `mtg_comprehensive_rules` - Magic rules text.
- `mtgdecks_articles` - MTGDecks strategy, theory, guide, and meta articles.
- `mtgdecks_meta_decks` - format/archetype/top deck snapshots.
- `scryfall_bulk` - Scryfall card text corpus for semantic card context.

## Embeddings

The schema expects 384-dimensional vectors by default. Use matching settings:

```bash
EMBEDDING_DIMENSIONS=384
```

OpenAI semantic embeddings:

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

Local semantic embeddings:

```bash
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

Hash embeddings are only for smoke tests and are not semantic.

## Card Corpus Formatting

`join_card_fields()` labels important Scryfall fields before embedding, including name, type, mana value, colors, color identity, keywords, legal formats, and oracle text. Re-ingest `scryfall_bulk` after changing this formatting so stored documents match the current representation.
