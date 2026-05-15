"""Import MTGDecks meta deck JSON into the RAG store."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.logging import configure_logging, log_extra
from app.rag.embeddings import get_embedding_provider
from app.rag.ingestion import load_mtgdecks_meta_decks_file
from app.rag.repository import RagRepository

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("data/mtgdecks-meta-decks.json"),
        help="Path to downloaded MTGDecks meta deck JSON.",
    )
    parser.add_argument("--batch-size", type=int, help="Embedding/upsert batch size.")
    parser.add_argument("--batch-delay-seconds", type=float, help="Delay between embedding batches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger.info(
        "Embedding configuration loaded",
        extra=log_extra(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        ),
    )
    if not args.path.exists():
        raise SystemExit(f"File does not exist: {args.path}")

    logger.info("Loading MTGDecks meta decks", extra=log_extra(path=str(args.path)))
    documents = load_mtgdecks_meta_decks_file(args.path)
    logger.info("Prepared MTGDecks meta deck RAG documents", extra=log_extra(document_count=len(documents)))
    with SessionLocal() as session:
        repository = RagRepository(session, get_embedding_provider())
        count = repository.upsert_many(
            documents,
            batch_size=args.batch_size or settings.rag_ingest_batch_size,
            batch_delay_seconds=(
                args.batch_delay_seconds
                if args.batch_delay_seconds is not None
                else settings.rag_ingest_batch_delay_seconds
            ),
        )

    logger.info("Imported MTGDecks meta deck RAG documents", extra=log_extra(imported=count, path=str(args.path)))


if __name__ == "__main__":
    main()
