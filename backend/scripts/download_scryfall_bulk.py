"""Download Scryfall's default-cards bulk JSON dump."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from app.core.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/scryfall-default-cards.json"),
        help="Destination path for the downloaded JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=settings.scryfall_api_base_url, timeout=60) as client:
        bulk_response = client.get("/bulk-data")
        bulk_response.raise_for_status()
        bulk_entries = bulk_response.json()["data"]

        default_cards = next(entry for entry in bulk_entries if entry["type"] == "default_cards")
        download_response = client.get(default_cards["download_uri"])
        download_response.raise_for_status()
        args.output.write_bytes(download_response.content)

    print(f"Downloaded {default_cards['name']} to {args.output}")


if __name__ == "__main__":
    main()
