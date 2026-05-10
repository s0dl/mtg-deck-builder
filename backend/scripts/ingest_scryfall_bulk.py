"""Import a local Scryfall bulk JSON dump into the pgvector RAG store."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.database import SessionLocal
from app.rag.embeddings import get_embedding_provider
from app.rag.ingestion import load_scryfall_bulk_file
from app.rag.repository import RagRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a downloaded Scryfall bulk JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.path.exists():
        raise SystemExit(f"File does not exist: {args.path}")

    documents = load_scryfall_bulk_file(args.path)
    with SessionLocal() as session:
        repository = RagRepository(session, get_embedding_provider())
        count = repository.upsert_many(documents)

    print(f"Imported {count} RAG documents from {args.path}")


if __name__ == "__main__":
    main()
