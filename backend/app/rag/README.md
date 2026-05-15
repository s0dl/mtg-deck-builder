# RAG Layer

The RAG layer provides long-lived rules, strategy, meta, and card-text context using Postgres with pgvector.

## Sources

- `mtg_comprehensive_rules` - Magic Comprehensive Rules chunks.
- `mtgdecks_articles` - MTGDecks strategy, theory, guide, and meta article chunks.
- `mtgdecks_meta_decks` - format archetype/top deck snapshots.
- `scryfall_bulk` - Scryfall card corpus records for semantic card context.

Live Scryfall remains the source of truth for current card data during deck generation. RAG card records help retrieval, but they should not replace live legality, price, and oracle lookups.

## Storage

The schema is in `database/init/001_pgvector.sql`.

`rag_documents` stores:

- `source` - provider name, such as `scryfall_bulk` or `mtgdecks_articles`.
- `source_id` - stable upstream identifier.
- `title` - human-readable label.
- `content` - embedded chunk text.
- `metadata` - source-specific filters and facts.
- `embedding` - pgvector embedding.

## Retrieval

`retriever.py` supports vector search, text fallback, and metadata filters. Vector search uses the configured embedding provider. Text fallback is intentionally basic and should be improved with ranked lexical search.

## Embeddings

OpenAI:

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=384
```

FastEmbed:

```bash
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSIONS=384
```

Hash embeddings are only for smoke tests.

## Commands

```bash
python -m scripts.download_mtg_rules --output data/mtg-comprehensive-rules.txt
python -m scripts.ingest_mtg_rules data/mtg-comprehensive-rules.txt

python -m scripts.download_mtgdecks_articles --pages 3 --output data/mtgdecks-articles.json
python -m scripts.ingest_mtgdecks_articles data/mtgdecks-articles.json

python -m scripts.download_mtgdecks_meta_decks --formats Standard Pioneer Modern Legacy Pauper Commander --top 30
python -m scripts.ingest_mtgdecks_meta_decks data/mtgdecks-meta-decks.json

python -m scripts.download_scryfall_card_corpus --output data/scryfall-card-corpus.json
python -m scripts.ingest_scryfall_card_corpus data/scryfall-card-corpus.json
```
