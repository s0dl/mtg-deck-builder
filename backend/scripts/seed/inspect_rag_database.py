"""Inspect indexed RAG database contents without calling external APIs."""

from __future__ import annotations

import argparse
import json
from typing import Any

from sqlalchemy import func, or_, select

from app.core.database import SessionLocal
from app.rag.documents import RagDocument

INGEST_SOURCES = (
    "scryfall_bulk",
    "mtgdecks_articles",
    "mtgdecks_meta_decks",
    "mtg_comprehensive_rules",
    "foundational_strategy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cards",
        nargs="*",
        default=["Lightning Bolt", "Consider", "Command Tower", "Steam Vents", "Island"],
        help="Exact card titles to inspect in the indexed Scryfall corpus.",
    )
    parser.add_argument(
        "--source",
        default="scryfall_bulk",
        help="RAG source to sample. Use 'all' to sample every source.",
    )
    parser.add_argument(
        "--check-ingest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show expected ingest source health for cards, articles, meta decks, rules, and seed strategy.",
    )
    parser.add_argument("--format", default="modern", help="Legality key to show for card rows.")
    parser.add_argument("--limit", type=int, default=12, help="Rows to sample per section.")
    parser.add_argument(
        "--query",
        default="aggro tempo prowess interaction",
        help="Terms used for relevant article/meta/rule samples.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as session:
        source_counts = session.execute(
            select(RagDocument.source, func.count()).group_by(RagDocument.source).order_by(RagDocument.source)
        ).all()
        count_lookup = {source: count for source, count in source_counts}
        card_rows = _card_rows(session, args.cards, args.format, args.limit)
        ingest_checks = _ingest_checks(session, count_lookup, args.format, min(args.limit, 5)) if args.check_ingest else []
        coverage = _coverage(session, args.format, args.query, min(args.limit, 8))
        samples = _sample_rows(session, args.source, args.format, args.limit)

    payload = {
        "source_counts": [{"source": source, "count": count} for source, count in source_counts],
        "ingest_checks": ingest_checks,
        "coverage": coverage,
        "card_rows": card_rows,
        "samples": samples,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("RAG source counts")
    print("=================")
    for row in payload["source_counts"]:
        print(f"{row['source']}: {row['count']}")

    if args.check_ingest:
        print("\nExpected ingest checks")
        print("======================")
        for check in ingest_checks:
            status = "ok" if check["present"] else "missing"
            print(f"\n{check['source']}: {status} ({check['count']} rows)")
            _print_rows(check["samples"])

    print(f"\nCoverage for {args.format}")
    print("====================" + "=" * len(args.format))
    _print_coverage(coverage)

    print("\nRequested card rows")
    print("===================")
    _print_rows(card_rows)

    print(f"\nSample rows ({args.source})")
    print("=================" + "=" * len(args.source))
    _print_rows(samples)


def _card_rows(session, card_names: list[str], mtg_format: str, limit: int) -> list[dict[str, Any]]:
    if not card_names:
        return []
    statement = (
        select(RagDocument)
        .where(RagDocument.source == "scryfall_bulk")
        .where(RagDocument.title.in_(card_names))
        .order_by(RagDocument.title, RagDocument.source_id)
        .limit(limit)
    )
    return [_document_summary(document, mtg_format) for document in session.scalars(statement).all()]


def _ingest_checks(
    session,
    count_lookup: dict[str, int],
    mtg_format: str,
    limit: int,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for source in INGEST_SOURCES:
        count = count_lookup.get(source, 0)
        checks.append(
            {
                "source": source,
                "count": count,
                "present": count > 0,
                "samples": _sample_rows(session, source, mtg_format, limit) if count > 0 else [],
            }
        )
    return checks


def _coverage(session, mtg_format: str, query: str, limit: int) -> dict[str, Any]:
    return {
        "article_format_counts": _metadata_value_counts(session, "mtgdecks_articles", "format"),
        "meta_deck_format_counts": _metadata_value_counts(session, "mtgdecks_meta_decks", "format"),
        "rules_prefix_counts": _rules_prefix_counts(session),
        "relevant_articles": _relevant_rows(
            session=session,
            source="mtgdecks_articles",
            mtg_format=mtg_format,
            query=query,
            limit=limit,
        ),
        "relevant_meta_decks": _relevant_rows(
            session=session,
            source="mtgdecks_meta_decks",
            mtg_format=mtg_format,
            query=query,
            limit=limit,
        ),
        "deck_construction_rules": _rule_rows(session, limit=limit),
    }


def _metadata_value_counts(session, source: str, metadata_key: str) -> list[dict[str, Any]]:
    value = RagDocument.metadata_[metadata_key].as_string()
    rows = session.execute(
        select(value, func.count())
        .where(RagDocument.source == source)
        .group_by(value)
        .order_by(value)
    ).all()
    return [{"value": key or "", "count": count} for key, count in rows]


def _rules_prefix_counts(session) -> list[dict[str, Any]]:
    rule_number = RagDocument.metadata_["rule_number"].as_string()
    prefix = func.split_part(rule_number, ".", 1)
    rows = session.execute(
        select(prefix, func.count())
        .where(RagDocument.source == "mtg_comprehensive_rules")
        .group_by(prefix)
        .order_by(prefix)
    ).all()
    return [{"prefix": key or "", "count": count} for key, count in rows[:20]]


def _relevant_rows(
    session,
    source: str,
    mtg_format: str,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    terms = [term for term in query.lower().split() if len(term) > 2]
    statement = select(RagDocument).where(RagDocument.source == source)
    if source in {"mtgdecks_articles", "mtgdecks_meta_decks"}:
        statement = statement.where(
            or_(
                RagDocument.metadata_["format"].as_string() == mtg_format,
                RagDocument.metadata_["format"].as_string() == "",
            )
        )
    if terms:
        statement = statement.where(
            or_(
                *[
                    or_(
                        RagDocument.title.ilike(f"%{term}%"),
                        RagDocument.content.ilike(f"%{term}%"),
                    )
                    for term in terms[:6]
                ]
            )
        )
    statement = statement.order_by(RagDocument.title, RagDocument.source_id).limit(limit)
    return [_document_summary(document, mtg_format) for document in session.scalars(statement).all()]


def _rule_rows(session, limit: int) -> list[dict[str, Any]]:
    prefixes = ("100.2", "100.4", "100.4a", "100.4b", "903.5", "903.6")
    rule_number = RagDocument.metadata_["rule_number"].as_string()
    statement = (
        select(RagDocument)
        .where(RagDocument.source == "mtg_comprehensive_rules")
        .where(or_(*[rule_number.like(f"{prefix}%") for prefix in prefixes]))
        .order_by(rule_number, RagDocument.source_id)
        .limit(limit)
    )
    return [_document_summary(document, "modern") for document in session.scalars(statement).all()]


def _sample_rows(session, source: str, mtg_format: str, limit: int) -> list[dict[str, Any]]:
    statement = select(RagDocument).order_by(RagDocument.source, RagDocument.title).limit(limit)
    if source != "all":
        statement = statement.where(RagDocument.source == source)
    return [_document_summary(document, mtg_format) for document in session.scalars(statement).all()]


def _document_summary(document: RagDocument, mtg_format: str) -> dict[str, Any]:
    metadata = document.metadata_ or {}
    legalities = metadata.get("legalities") if isinstance(metadata.get("legalities"), dict) else {}
    return {
        "source": document.source,
        "source_id": document.source_id,
        "title": document.title,
        "kind": metadata.get("kind"),
        "type_line": metadata.get("type_line"),
        "mana_value": metadata.get("mana_value"),
        "colors": metadata.get("colors"),
        "color_identity": metadata.get("color_identity"),
        f"{mtg_format}_legality": legalities.get(mtg_format),
        "format": metadata.get("format"),
        "section": metadata.get("section"),
        "rule_number": metadata.get("rule_number"),
        "archetype": metadata.get("archetype"),
        "metagame_share": metadata.get("metagame_share"),
        "url": metadata.get("url"),
        "top_deck_url": metadata.get("top_deck_url"),
        "deck_cards": _deck_cards_preview(metadata),
        "price_usd": _price_usd(metadata, document.content),
        "content_preview": " ".join(document.content.split())[:180],
    }


def _deck_cards_preview(metadata: dict[str, Any], limit: int = 20) -> list[str] | None:
    top_deck_cards = metadata.get("top_deck_cards")
    if isinstance(top_deck_cards, list) and top_deck_cards:
        return [
            f"{card.get('count')} {card.get('name')}"
            for card in top_deck_cards[:limit]
            if isinstance(card, dict) and card.get("count") and card.get("name")
        ]
    card_names = metadata.get("card_names")
    if isinstance(card_names, list) and card_names:
        return [str(name) for name in card_names[:limit] if name]
    return None


def _price_usd(metadata: dict[str, Any], content: str) -> float | None:
    prices = metadata.get("prices")
    if isinstance(prices, dict):
        parsed = _parse_float(prices.get("usd"))
        if parsed is not None:
            return parsed
    parsed = _parse_float(metadata.get("estimated_price_usd"))
    if parsed is not None:
        return parsed
    marker = "Price USD:"
    if marker in content:
        value = content.split(marker, 1)[1].split(maxsplit=1)[0]
        return _parse_float(value)
    return None


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _print_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(no rows)")
        return
    for row in rows:
        print(json.dumps(row, sort_keys=True))


def _print_coverage(coverage: dict[str, Any]) -> None:
    print("\nArticle format counts")
    _print_rows(coverage["article_format_counts"])
    print("\nMeta deck format counts")
    _print_rows(coverage["meta_deck_format_counts"])
    print("\nRules prefix counts")
    _print_rows(coverage["rules_prefix_counts"])
    print("\nRelevant articles")
    _print_rows(coverage["relevant_articles"])
    print("\nRelevant meta decks")
    _print_rows(coverage["relevant_meta_decks"])
    print("\nDeck construction rules")
    _print_rows(coverage["deck_construction_rules"])


if __name__ == "__main__":
    main()
