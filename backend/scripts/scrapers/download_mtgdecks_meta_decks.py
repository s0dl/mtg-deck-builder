"""Download MTGDecks format archetype and top decklist snapshots.

Robots note:
The supplied robots.txt disallows several AI crawler user agents, including ChatGPT-User and
GPTBot. This script is intended for the project operator's own runs with an identifying user agent,
low request rate, and only paths allowed for User-agent: *.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from app.core.config import get_settings
from app.core.logging import configure_logging, log_extra

BASE_URL = "https://mtgdecks.net"
DEFAULT_FORMATS = ("Standard", "Pioneer", "Modern", "Legacy", "Pauper", "Commander")
ARCHETYPE_PERCENT_PATTERN = re.compile(r"(?P<percent>\d+(?:\.\d+)?)\s*%")
DECKLIST_PATH_PATTERN = re.compile(
    r"^/(?P<format>Standard|Pioneer|Modern|Legacy|Pauper|Commander)/"
    r"(?P<slug>[^/?#]+)-decklist-by-[^/?#]+-\d+/?$",
    re.IGNORECASE,
)
CARD_LINE_PATTERN = re.compile(r"^(?P<count>\d+)\s+(?P<name>[^#]+?)(?:\s+\$?[0-9].*)?$")
COUNT_PATTERN = re.compile(r"^\d+$")
DECK_SECTION_START_PATTERN = re.compile(r"^Maindeck(?:\s*\(\d+\))?$", re.IGNORECASE)
DECK_SECTION_ENDINGS = (
    "buy this deck",
    "deck tools",
    "export & save",
    "embedding code",
    "if you find any error",
    "suggest archetype",
    "last update",
)
NON_ARCHETYPE_SLUGS = {
    "analysis",
    "budget",
    "decks",
    "events",
    "format-staples",
    "metagame",
    "price",
    "staples",
    "tournaments",
    "winrates",
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinkRecord:
    href: str
    text: str
    index: int


class VisibleTextParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[LinkRecord] = []
        self.texts: list[str] = []
        self._ignored_depth = 0
        self._current_href: str | None = None
        self._current_link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._current_href = urljoin(self.base_url, href)
                self._current_link_text = []

    def handle_endtag(self, tag: str) -> None:
        if self._ignored_depth:
            if tag in {"script", "style", "noscript", "svg"}:
                self._ignored_depth -= 1
            return
        if tag == "a" and self._current_href:
            self.links.append(
                LinkRecord(
                    href=self._current_href,
                    text=clean_text(" ".join(self._current_link_text)),
                    index=len(self.texts),
                )
            )
            self._current_href = None
            self._current_link_text = []

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = clean_text(data)
        if not text:
            return
        self.texts.append(text)
        if self._current_href is not None:
            self._current_link_text.append(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/mtgdecks-meta-decks.json"))
    parser.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), choices=list(DEFAULT_FORMATS))
    parser.add_argument("--top", type=int, default=10, help="Maximum archetypes per format.")
    parser.add_argument("--delay-seconds", type=float, default=1.5)
    parser.add_argument(
        "--user-agent",
        default="mtg-deck-builder/0.1 (+https://github.com/local/mtg-deck-builder)",
    )
    return parser.parse_args()


def format_url(mtg_format: str) -> str:
    return f"{BASE_URL}/{mtg_format}"


def parse_format_page(page_html: str, page_url: str, mtg_format: str, top: int) -> list[dict[str, object]]:
    parser = VisibleTextParser(page_url)
    parser.feed(page_html)
    archetypes: list[dict[str, object]] = []
    seen: set[str] = set()
    prefix = f"/{mtg_format}/".lower()

    for link in parser.links:
        path = urlparse(link.href).path
        if not path.lower().startswith(prefix):
            continue
        slug = path.rstrip("/").rsplit("/", 1)[-1].lower()
        if slug in NON_ARCHETYPE_SLUGS:
            continue
        if ":" in slug:
            continue
        if DECKLIST_PATH_PATTERN.match(path):
            continue
        if path.rstrip("/").count("/") != 2:
            continue
        if link.href in seen:
            continue
        seen.add(link.href)
        archetypes.append(
            {
                "format": mtg_format.lower(),
                "name": link.text or path.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title(),
                "url": link.href,
                "metagame_share": nearby_percentage(parser.texts, link.index),
            }
        )
        if len(archetypes) >= top:
            break

    return archetypes


def parse_archetype_page(page_html: str, page_url: str, mtg_format: str) -> str:
    parser = VisibleTextParser(page_url)
    parser.feed(page_html)
    for link in parser.links:
        path = urlparse(link.href).path
        match = DECKLIST_PATH_PATTERN.match(path)
        if match and match.group("format").lower() == mtg_format.lower():
            return link.href
    return ""


def parse_decklist_page(page_html: str, page_url: str) -> dict[str, object]:
    parser = VisibleTextParser(page_url)
    parser.feed(page_html)
    cards: list[dict[str, object]] = []
    seen_lines: set[str] = set()
    start_index, end_index = deck_text_bounds(parser.texts)
    deck_texts = parser.texts[start_index:end_index]

    for text in deck_texts:
        if text in seen_lines:
            continue
        seen_lines.add(text)
        match = CARD_LINE_PATTERN.match(text)
        if not match:
            continue
        count = int(match.group("count"))
        name = clean_card_name(match.group("name"))
        if count <= 0 or not name or should_skip_card_line(name):
            continue
        cards.append({"count": count, "name": name})

    if not cards:
        cards.extend(cards_from_quantity_links(parser.links, parser.texts, start_index, end_index))

    return {"url": page_url, "cards": cards}


def deck_text_bounds(texts: list[str]) -> tuple[int, int]:
    start = len(texts)
    for index, text in enumerate(texts):
        if DECK_SECTION_START_PATTERN.match(text):
            start = index + 1
            break

    end = len(texts)
    for index in range(start, len(texts)):
        lowered = texts[index].lower()
        if any(lowered.startswith(ending) for ending in DECK_SECTION_ENDINGS):
            end = index
            break

    return start, end


def cards_from_quantity_links(
    links: list[LinkRecord],
    texts: list[str],
    start_index: int,
    end_index: int,
) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    for link in links:
        if link.index < start_index or link.index >= end_index:
            continue
        card_text_index = find_link_text_index(texts, link.text, start_index, link.index)
        if card_text_index is None or card_text_index == 0:
            continue
        count_text = texts[card_text_index - 1]
        if not COUNT_PATTERN.match(count_text):
            continue
        name = clean_card_name(link.text)
        if not name or should_skip_card_line(name):
            continue
        cards.append({"count": int(count_text), "name": name})
    return cards


def find_link_text_index(
    texts: list[str],
    link_text: str,
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(min(end_index, len(texts)) - 1, start_index - 1, -1):
        if texts[index] == link_text:
            return index
    return None


def nearby_percentage(texts: list[str], link_index: int) -> float | None:
    start = max(link_index - 4, 0)
    end = min(link_index + 8, len(texts))
    for text in texts[start:end]:
        match = ARCHETYPE_PERCENT_PATTERN.search(text)
        if match:
            return float(match.group("percent"))
    return None


def clean_text(text: str) -> str:
    return " ".join(text.split())


def clean_card_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip(" -\t")


def should_skip_card_line(name: str) -> bool:
    lowered = name.lower()
    return lowered in {"maindeck", "sideboard", "creatures", "spells", "lands"} or len(name) < 2


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    headers = {"User-Agent": args.user_agent}
    with httpx.Client(timeout=60, follow_redirects=True, headers=headers) as client:
        for mtg_format in args.formats:
            logger.info("Fetching MTGDecks format page", extra=log_extra(format=mtg_format))
            response = client.get(format_url(mtg_format))
            response.raise_for_status()
            archetypes = parse_format_page(response.text, str(response.url), mtg_format, args.top)

            for archetype in archetypes:
                time.sleep(args.delay_seconds)
                logger.info(
                    "Fetching MTGDecks archetype page",
                    extra=log_extra(format=mtg_format, archetype=archetype["name"]),
                )
                archetype_response = client.get(str(archetype["url"]))
                if archetype_response.status_code in {403, 404}:
                    logger.warning(
                        "Skipping inaccessible MTGDecks archetype page",
                        extra=log_extra(
                            format=mtg_format,
                            archetype=archetype["name"],
                            status_code=archetype_response.status_code,
                            url=str(archetype["url"]),
                        ),
                    )
                    continue
                archetype_response.raise_for_status()
                decklist_url = parse_archetype_page(
                    archetype_response.text,
                    str(archetype_response.url),
                    mtg_format,
                )
                if not decklist_url:
                    records.append({**archetype, "top_deck": None})
                    continue

                time.sleep(args.delay_seconds)
                logger.info(
                    "Fetching MTGDecks top decklist",
                    extra=log_extra(format=mtg_format, archetype=archetype["name"], url=decklist_url),
                )
                decklist_response = client.get(decklist_url)
                if decklist_response.status_code in {403, 404}:
                    logger.warning(
                        "Skipping inaccessible MTGDecks decklist page",
                        extra=log_extra(
                            format=mtg_format,
                            archetype=archetype["name"],
                            status_code=decklist_response.status_code,
                            url=decklist_url,
                        ),
                    )
                    records.append({**archetype, "top_deck": None})
                    continue
                decklist_response.raise_for_status()
                top_deck = parse_decklist_page(decklist_response.text, str(decklist_response.url))
                records.append({**archetype, "top_deck": top_deck})

    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Downloaded MTGDecks meta deck records", extra=log_extra(count=len(records), output=str(args.output)))


if __name__ == "__main__":
    main()
