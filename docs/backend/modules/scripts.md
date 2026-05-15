# Scripts Module

Path: `scripts`

## Purpose

Download and ingest external knowledge sources into local files and `rag_documents`.

## Download Scripts

```bash
python -m scripts.download_mtg_rules --output data/mtg-comprehensive-rules.txt
python -m scripts.download_mtgdecks_articles --pages 3 --output data/mtgdecks-articles.json
python -m scripts.download_mtgdecks_meta_decks --formats Standard Pioneer Modern Legacy Pauper Commander --top 30
python -m scripts.download_scryfall_card_corpus --output data/scryfall-card-corpus.json
```

## Ingest Scripts

```bash
python -m scripts.ingest_mtg_rules data/mtg-comprehensive-rules.txt
python -m scripts.ingest_mtgdecks_articles data/mtgdecks-articles.json
python -m scripts.ingest_mtgdecks_meta_decks data/mtgdecks-meta-decks.json
python -m scripts.ingest_scryfall_card_corpus data/scryfall-card-corpus.json
```

## Operational Notes

- Run scripts from an environment that can reach `DATABASE_URL`.
- Use Docker service hostnames only inside Docker.
- Use `--batch-size` and `--batch-delay-seconds` on ingest scripts when provider rate limits are a concern.
- Upserts use stable source/source-id identities, so rerunning ingestion updates existing documents.
- If source formatting changes substantially, deleting rows for that source before re-ingesting is cleaner than relying only on upsert.
