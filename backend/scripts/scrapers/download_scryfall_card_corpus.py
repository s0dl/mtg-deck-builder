"""Download the Scryfall default-card bulk corpus for RAG card-text indexing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.logging import configure_logging, log_extra

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/scryfall-card-corpus.json"),
        help="Destination path for Scryfall default-card JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        logger.info("Fetching Scryfall bulk data index")
        bulk_response = client.get(f"{settings.scryfall_api_base_url.rstrip('/')}/bulk-data")
        bulk_response.raise_for_status()
        download_uri = default_cards_download_uri(bulk_response.json())
        logger.info("Fetching Scryfall default-card corpus", extra=log_extra(download_uri=download_uri))
        card_response = client.get(download_uri)
        card_response.raise_for_status()

    args.output.write_bytes(card_response.content)
    logger.info(
        "Downloaded Scryfall card corpus",
        extra=log_extra(output=str(args.output), bytes=len(card_response.content)),
    )


def default_cards_download_uri(payload: dict) -> str:
    for item in payload.get("data", []):
        if item.get("type") == "default_cards" and item.get("download_uri"):
            return str(item["download_uri"])
    raise ValueError("Scryfall bulk data response did not include default_cards download_uri.")


if __name__ == "__main__":
    main()
