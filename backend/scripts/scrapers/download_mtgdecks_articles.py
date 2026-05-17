"""Download MTGDecks strategy, guide, and meta article data from paginated listings."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from app.core.config import get_settings
from app.core.logging import configure_logging, log_extra

DEFAULT_SECTION_URLS = {
    "theory": "https://mtgdecks.net/theory",
    "guides": "https://mtgdecks.net/guides",
    "meta": "https://mtgdecks.net/meta",
}
DEFAULT_PAGE_COUNT = 3
MTG_FORMATS = ("standard", "pioneer", "modern", "legacy", "vintage", "commander", "pauper")
ARTICLE_PATH_PATTERN = re.compile(r"^/(?P<section>theory|guides|meta)/[^/?#]+-mtg-\d+/?$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_READ_TIME_PATTERN = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})\s*·\s*(?P<read_time>\d+ min read)$")
PUBLISHED_PATTERN = re.compile(r"^Published:\s*(?P<date>\d{4}-\d{2}-\d{2})")
STOP_TEXT_PREFIXES = (
    "Sign Up for MTGDecks newsletter",
    "Please enable JavaScript",
    "Upgrade your account",
    "Privacy Policy",
    "Terms",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextChunk:
    text: str
    tag: str


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
        self.chunks: list[TextChunk] = []
        self._tag_stack: list[str] = []
        self._ignored_depth = 0
        self._current_href: str | None = None
        self._current_link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tag_stack.append(tag)
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
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
            if self._tag_stack:
                self._tag_stack.pop()
            return
        if tag == "a" and self._current_href is not None:
            text = clean_text(" ".join(self._current_link_text))
            self.links.append(LinkRecord(self._current_href, text, len(self.chunks)))
            self._current_href = None
            self._current_link_text = []
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = clean_text(data)
        if not text:
            return
        tag = self._tag_stack[-1] if self._tag_stack else ""
        self.chunks.append(TextChunk(text=text, tag=tag))
        if self._current_href is not None:
            self._current_link_text.append(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mtgdecks-articles.json"),
        help="Destination path for downloaded article JSON.",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=sorted(DEFAULT_SECTION_URLS),
        default=list(DEFAULT_SECTION_URLS),
        help="MTGDecks sections to scan.",
    )
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGE_COUNT, help="Pages to scan per section.")
    return parser.parse_args()


def listing_url(section: str, page: int) -> str:
    if page <= 1:
        return DEFAULT_SECTION_URLS[section]
    return f"https://mtgdecks.net/articles/index/{section}/page:{page}"


def parse_listing(page_html: str, page_url: str, section: str) -> list[dict[str, str]]:
    parser = VisibleTextParser(page_url)
    parser.feed(page_html)

    articles: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for link in parser.links:
        path = urlparse(link.href).path
        match = ARTICLE_PATH_PATTERN.match(path)
        if not match or match.group("section") != section or link.href in seen_urls:
            continue

        title = link.text or nearest_title_after_link(parser.chunks, link.index)
        if not title or title.lower() in {section, "read more"}:
            continue

        articles.append(
            {
                "source": "mtgdecks_articles",
                "section": section,
                "url": link.href,
                "title": title,
            }
        )
        seen_urls.add(link.href)

    return articles


def parse_article(article_html: str, article_url: str, section: str) -> dict[str, str | list[str]]:
    parser = VisibleTextParser(article_url)
    parser.feed(article_html)
    texts = [chunk.text for chunk in parser.chunks]
    title = first_heading(parser.chunks) or title_from_url(article_url)
    title_index = find_text_index(texts, title)
    content_chunks = article_content_chunks(texts, title_index)

    return {
        "source": "mtgdecks_articles",
        "section": section,
        "url": article_url,
        "title": title,
        "category": section.title(),
        "format": infer_format(title, content_chunks),
        "summary": first_long_text(content_chunks),
        "published_at": article_date(texts),
        "read_time": article_read_time(texts),
        "author": nearest_author(texts, title_index),
        "content": "\n".join(content_chunks),
        "content_chunks": content_chunks,
    }


def clean_text(text: str) -> str:
    return " ".join(text.split())


def nearest_title_after_link(chunks: list[TextChunk], link_index: int) -> str:
    for chunk in chunks[link_index : min(link_index + 5, len(chunks))]:
        if len(chunk.text) > 8 and chunk.tag in {"h1", "h2", "h3", "h4", "a"}:
            return chunk.text
    return ""


def first_heading(chunks: list[TextChunk]) -> str:
    for chunk in chunks:
        if chunk.tag == "h1" and len(chunk.text) > 8:
            return chunk.text
    for chunk in chunks:
        if chunk.tag in {"h2", "h3"} and len(chunk.text) > 8:
            return chunk.text
    return ""


def title_from_url(url: str) -> str:
    slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"-mtg-\d+$", "", slug)
    return " ".join(word.capitalize() for word in slug.split("-"))


def find_text_index(texts: list[str], needle: str) -> int:
    try:
        return texts.index(needle)
    except ValueError:
        return -1


def article_content_chunks(texts: list[str], title_index: int) -> list[str]:
    start = max(title_index + 1, 0)
    chunks: list[str] = []
    seen: set[str] = set()
    for text in texts[start:]:
        if should_stop_content(text):
            break
        if should_skip_content(text):
            continue
        if text in seen:
            continue
        chunks.append(text)
        seen.add(text)
    return chunks


def should_stop_content(text: str) -> bool:
    return any(text.startswith(prefix) for prefix in STOP_TEXT_PREFIXES)


def should_skip_content(text: str) -> bool:
    if text in {"MTG", "Metagame", "Decks", "Articles", "Login", "Register", "Home"}:
        return True
    if text.startswith("https://"):
        return True
    return False


def first_long_text(texts: list[str]) -> str:
    for text in texts:
        if len(text) >= 80:
            return text
    return ""


def infer_format(title: str, content_chunks: list[str]) -> str:
    searchable = f"{title} {' '.join(content_chunks[:4])}".lower()
    for mtg_format in MTG_FORMATS:
        if re.search(rf"\b{re.escape(mtg_format)}\b", searchable):
            return mtg_format
    return ""


def article_date(texts: list[str]) -> str:
    for text in texts:
        match = DATE_READ_TIME_PATTERN.match(text)
        if match:
            return match.group("date")
        match = PUBLISHED_PATTERN.match(text)
        if match:
            return match.group("date")
        if DATE_PATTERN.match(text):
            return text
    return ""


def article_read_time(texts: list[str]) -> str:
    for text in texts:
        match = DATE_READ_TIME_PATTERN.match(text)
        if match:
            return match.group("read_time")
    return ""


def nearest_author(texts: list[str], title_index: int) -> str:
    if title_index < 0:
        return ""
    for text in texts[title_index + 1 : min(title_index + 8, len(texts))]:
        if text.lower().startswith("by "):
            return text[3:].strip()
        if DATE_READ_TIME_PATTERN.match(text) or DATE_PATTERN.match(text):
            continue
        if text.lower() in {"guides", "meta", "theory"}:
            continue
        if len(text) > 2:
            return text
    return ""


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str | list[str]]] = []
    seen_urls: set[str] = set()
    headers = {"User-Agent": "mtg-deck-builder/0.1 (+strategy indexer)"}
    with httpx.Client(timeout=60, follow_redirects=True, headers=headers) as client:
        for section in args.sections:
            for page in range(1, args.pages + 1):
                section_url = listing_url(section, page)
                logger.info(
                    "Fetching MTGDecks listing",
                    extra=log_extra(
                        section=section,
                        page=page,
                        total_pages=args.pages,
                        remaining_pages=args.pages - page,
                        url=section_url,
                    ),
                )
                listing_response = client.get(section_url)
                listing_response.raise_for_status()
                article_links = parse_listing(listing_response.text, str(listing_response.url), section)
                logger.info(
                    "Parsed MTGDecks listing",
                    extra=log_extra(section=section, page=page, article_count=len(article_links)),
                )
                for index, article in enumerate(article_links, start=1):
                    if article["url"] in seen_urls:
                        continue
                    logger.info(
                        "Fetching MTGDecks article",
                        extra=log_extra(
                            section=section,
                            page=page,
                            article_number=index,
                            total_articles=len(article_links),
                            remaining_articles=len(article_links) - index,
                            url=article["url"],
                        ),
                    )
                    article_response = client.get(article["url"])
                    article_response.raise_for_status()
                    record = parse_article(article_response.text, str(article_response.url), section)
                    if not record["title"]:
                        record["title"] = article["title"]
                    records.append(record)
                    seen_urls.add(str(record["url"]))

    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info(
        "Downloaded MTGDecks article records",
        extra=log_extra(article_count=len(records), output=str(args.output)),
    )


if __name__ == "__main__":
    main()
