"""Download Magic: The Gathering Comprehensive Rules text from Wizards."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import httpx

from app.core.config import get_settings
from app.core.logging import configure_logging, log_extra

RULES_PAGE_URL = "https://magic.wizards.com/en/rules"
TXT_LINK_PATTERN = re.compile(r'href="(?P<href>[^"]+MagicCompRules[^"]+\.txt)"', re.IGNORECASE)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mtg-comprehensive-rules.txt"),
        help="Destination path for the downloaded rules text file.",
    )
    parser.add_argument(
        "--rules-page-url",
        default=RULES_PAGE_URL,
        help="Wizards rules page to scan for the current TXT download link.",
    )
    return parser.parse_args()


def find_rules_txt_url(page_html: str, page_url: str) -> str:
    match = TXT_LINK_PATTERN.search(page_html)
    if not match:
        raise RuntimeError("Could not find a Comprehensive Rules TXT link on the rules page.")
    return urljoin(page_url, match.group("href"))


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        logger.info("Fetching Wizards rules page", extra=log_extra(url=args.rules_page_url))
        page_response = client.get(args.rules_page_url)
        page_response.raise_for_status()
        rules_txt_url = find_rules_txt_url(page_response.text, str(page_response.url))

        logger.info("Downloading Comprehensive Rules text", extra=log_extra(url=rules_txt_url))
        rules_response = client.get(rules_txt_url)
        rules_response.raise_for_status()
        args.output.write_bytes(rules_response.content)

    logger.info(
        "Downloaded Magic Comprehensive Rules",
        extra=log_extra(output=str(args.output), bytes=args.output.stat().st_size),
    )


if __name__ == "__main__":
    main()
