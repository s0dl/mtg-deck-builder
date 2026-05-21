from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.rag.chunking import chunk_text, join_card_fields
from app.core.logging import log_extra
from app.rag.repository import RagDocumentInput

logger = logging.getLogger(__name__)


def build_card_documents(card: dict[str, Any]) -> list[RagDocumentInput]:
    oracle_id = card.get("oracle_id") or card.get("id")
    name = card.get("name", "Unknown Card")
    content = join_card_fields(card)
    chunks = chunk_text(content)
    documents: list[RagDocumentInput] = []

    for index, chunk in enumerate(chunks):
        documents.append(
            RagDocumentInput(
                source="scryfall_bulk",
                source_id=f"{oracle_id}:{index}",
                title=name,
                content=chunk,
                metadata={
                    "kind": "card_text",
                    "card_id": card.get("id"),
                    "oracle_id": oracle_id,
                    "name": name,
                    "colors": card.get("colors", []),
                    "color_identity": card.get("color_identity", []),
                    "mana_value": card.get("cmc"),
                    "type_line": card.get("type_line"),
                    "keywords": card.get("keywords", []),
                    "legalities": card.get("legalities", {}),
                    "prices": card.get("prices", {}),
                    "price_usd": _price_usd(card),
                    "estimated_price_usd": _price_usd(card),
                    "layout": card.get("layout"),
                    "games": card.get("games", []),
                },
            )
        )

    return documents


def load_scryfall_card_corpus_file(path: Path) -> list[RagDocumentInput]:
    return load_scryfall_bulk_file(path)


def _price_usd(card: dict[str, Any]) -> float | None:
    prices = card.get("prices")
    if not isinstance(prices, dict):
        return None
    value = prices.get("usd")
    if value not in (None, ""):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    for key in ("usd_foil", "usd_etched"):
        value = prices.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def load_scryfall_bulk_file(path: Path) -> list[RagDocumentInput]:
    with path.open("r", encoding="utf-8") as handle:
        cards = json.load(handle)

    logger.info("Parsed Scryfall bulk JSON", extra=log_extra(card_count=len(cards), path=str(path)))
    documents: list[RagDocumentInput] = []
    skipped = 0
    for card in cards:
        if card.get("layout") in {"art_series", "emblem", "token", "double_faced_token"}:
            skipped += 1
            continue
        if "paper" not in card.get("games", []):
            skipped += 1
            continue
        documents.extend(build_card_documents(card))

    logger.info(
        "Built Scryfall RAG documents",
        extra=log_extra(document_count=len(documents), skipped_card_count=skipped),
    )
    return documents


RULE_HEADING_PATTERN = re.compile(r"(?m)^(?P<number>\d{3}(?:\.\d+[a-z]?)?)\. (?P<title>.+)$")


def load_comprehensive_rules_file(path: Path) -> list[RagDocumentInput]:
    content = path.read_text(encoding="utf-8-sig")
    matches = list(RULE_HEADING_PATTERN.finditer(content))
    logger.info(
        "Parsed MTG comprehensive rules text",
        extra=log_extra(path=str(path), heading_count=len(matches), bytes=len(content.encode("utf-8"))),
    )
    if not matches:
        return build_strategy_document(
            source="mtg_comprehensive_rules",
            source_id=path.stem,
            title="Magic: The Gathering Comprehensive Rules",
            content=content,
            metadata={"kind": "rules"},
        )

    documents: list[RagDocumentInput] = []
    for index, match in enumerate(matches):
        rule_number = match.group("number")
        rule_title = match.group("title").strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        rule_content = content[start:end].strip()

        for chunk_index, chunk in enumerate(chunk_text(rule_content, max_words=220, overlap_words=30)):
            documents.append(
                RagDocumentInput(
                    source="mtg_comprehensive_rules",
                    source_id=f"{rule_number}:{chunk_index}",
                    title=f"Rule {rule_number}. {rule_title}",
                    content=chunk,
                    metadata={
                        "kind": "rules",
                        "rule_number": rule_number,
                        "rule_title": rule_title,
                    },
                )
            )

    logger.info("Built MTG rules RAG documents", extra=log_extra(document_count=len(documents)))
    return documents


def load_strategy_article_index_file(path: Path) -> list[RagDocumentInput]:
    with path.open("r", encoding="utf-8") as handle:
        articles = json.load(handle)

    logger.info("Parsed strategy article metadata", extra=log_extra(article_count=len(articles), path=str(path)))
    documents: list[RagDocumentInput] = []
    for article in articles:
        title = article.get("title")
        url = article.get("url")
        if not title or not url:
            continue

        summary = article.get("summary") or ""
        content = "\n".join(
            field
            for field in [
                title,
                article.get("category"),
                article.get("section"),
                article.get("format"),
                article.get("game"),
                article.get("read_time"),
                summary,
                article.get("content"),
                article.get("published_at"),
                article.get("author"),
                url,
            ]
            if field
        )
        for chunk_index, chunk in enumerate(chunk_text(content, max_words=260, overlap_words=40)):
            documents.append(
                RagDocumentInput(
                    source=article.get("source", "strategy_articles"),
                    source_id=f"{url}:{chunk_index}",
                    title=title,
                    content=chunk,
                    metadata={
                        "kind": "strategy_article_index",
                        "category": article.get("category"),
                        "author": article.get("author"),
                        "published_at": article.get("published_at"),
                        "url": url,
                        "section": article.get("section"),
                        "format": article.get("format"),
                        "game": article.get("game"),
                        "read_time": article.get("read_time"),
                        "chunk_index": chunk_index,
                    },
                )
            )

    logger.info("Built strategy article RAG documents", extra=log_extra(document_count=len(documents)))
    return documents


def load_mtgdecks_meta_decks_file(path: Path) -> list[RagDocumentInput]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    logger.info("Parsed MTGDecks meta deck records", extra=log_extra(record_count=len(records), path=str(path)))
    documents: list[RagDocumentInput] = []
    skipped_without_cards = 0
    for record in records:
        mtg_format = record.get("format")
        name = record.get("name")
        url = record.get("url")
        if not mtg_format or not name or not url:
            continue

        top_deck = record.get("top_deck") if isinstance(record.get("top_deck"), dict) else {}
        cards = top_deck.get("cards", []) if isinstance(top_deck, dict) else []
        if not cards:
            skipped_without_cards += 1
            continue
        card_lines = [
            f"{card.get('count')} {card.get('name')}"
            for card in cards
            if isinstance(card, dict) and card.get("count") and card.get("name")
        ]
        if not card_lines:
            skipped_without_cards += 1
            continue
        content = "\n".join(
            field
            for field in [
                f"{name} in {mtg_format}",
                f"Format: {mtg_format}",
                f"Metagame share: {record.get('metagame_share')}%",
                f"Archetype URL: {url}",
                f"Top deck URL: {top_deck.get('url') if isinstance(top_deck, dict) else ''}",
                "Top deck cards:",
                "\n".join(card_lines),
            ]
            if field
        )

        for chunk_index, chunk in enumerate(chunk_text(content, max_words=260, overlap_words=40)):
            documents.append(
                RagDocumentInput(
                    source="mtgdecks_meta_decks",
                    source_id=f"{url}:{chunk_index}",
                    title=f"{name} ({str(mtg_format).title()})",
                    content=chunk,
                    metadata={
                        "kind": "meta_deck",
                        "format": mtg_format,
                        "archetype": name,
                        "metagame_share": record.get("metagame_share"),
                        "url": url,
                        "top_deck_url": top_deck.get("url") if isinstance(top_deck, dict) else "",
                        "top_deck_cards": [
                            {"count": card.get("count"), "name": card.get("name")}
                            for card in cards
                            if isinstance(card, dict) and card.get("count") and card.get("name")
                        ],
                        "card_names": [
                            card.get("name")
                            for card in cards
                            if isinstance(card, dict) and card.get("name")
                        ],
                        "chunk_index": chunk_index,
                    },
                )
            )

    logger.info(
        "Built MTGDecks meta deck RAG documents",
        extra=log_extra(document_count=len(documents), skipped_without_cards=skipped_without_cards),
    )
    return documents


def build_strategy_document(
    source: str,
    source_id: str,
    title: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> list[RagDocumentInput]:
    return [
        RagDocumentInput(
            source=source,
            source_id=f"{source_id}:{index}",
            title=title,
            content=chunk,
            metadata=metadata or {},
        )
        for index, chunk in enumerate(chunk_text(content))
    ]
