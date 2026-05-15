import logging
from typing import Any

from fastapi import APIRouter, Depends
import httpx
from sqlalchemy.orm import Session

from app.agent.deck_builder import OllamaDeckAgent, OpenAIDeckAgent
from app.agent.tools import DeckAgentTools
from app.core.config import get_settings
from app.core.database import get_session
from app.core.logging import log_extra
from app.llm.deck_builder import OpenAIDeckBuilder
from app.mcp.scryfall_client import ScryfallClient, scryfall_card_to_document
from app.models.deck import DeckRequest, DeckResponse, Format
from app.rag.retriever import RagRetriever, RetrievedDocument
from app.skills.deck_rules import validate_deck
from app.skills.deck_evaluation import rank_candidate_documents
from app.skills.mana_curve import calculate_mana_curve

router = APIRouter(prefix="/decks", tags=["decks"])
logger = logging.getLogger(__name__)

COLOR_TO_BASIC_LAND = {
    "W": "Plains",
    "U": "Island",
    "B": "Swamp",
    "R": "Mountain",
    "G": "Forest",
}

COLOR_TO_NAME = {
    "W": "white",
    "U": "blue",
    "B": "black",
    "R": "red",
    "G": "green",
}

COLOR_PAIR_NAMES = {
    frozenset({"W", "U"}): "azorius",
    frozenset({"U", "B"}): "dimir",
    frozenset({"B", "R"}): "rakdos",
    frozenset({"R", "G"}): "gruul",
    frozenset({"G", "W"}): "selesnya",
    frozenset({"W", "B"}): "orzhov",
    frozenset({"U", "R"}): "izzet",
    frozenset({"B", "G"}): "golgari",
    frozenset({"R", "W"}): "boros",
    frozenset({"G", "U"}): "simic",
}

NON_DECK_TYPE_MARKERS = {
    "Attraction",
    "Card",
    "Conspiracy",
    "Dungeon",
    "Emblem",
    "Phenomenon",
    "Plane",
    "Scheme",
    "Sticker",
    "Token",
    "Vanguard",
}

PLAYSTYLE_QUERY_TERMS = {
    "aggro": "creature haste prowess damage attack",
    "tempo": "flash flying counter draw bounce prowess",
    "control": "counter destroy exile draw sweeper removal",
    "combo": "search draw copy mana untap sacrifice",
    "midrange": "creature removal draw value planeswalker",
    "ramp": "search land mana add creature",
}

LIVE_CARD_SOURCES = {"scryfall_live", "scryfall_bulk"}


def _is_basic_land(card_name: str) -> bool:
    return card_name in {*COLOR_TO_BASIC_LAND.values(), "Wastes"}


def _is_land_card(card: dict) -> bool:
    name = str(card.get("name") or "")
    role = str(card.get("role") or "")
    type_line = str(card.get("type_line") or "")
    return _is_basic_land(name) or "Land" in role.split(" ") or "Land" in type_line.split(" ")


def _is_non_deck_game_object(document: RetrievedDocument) -> bool:
    type_line = document.metadata.get("type_line") or ""
    return any(marker in type_line.split(" ") for marker in NON_DECK_TYPE_MARKERS)


def _is_land_document(document: RetrievedDocument) -> bool:
    type_line = document.metadata.get("type_line") or ""
    return "Land" in type_line.split(" ")


def _is_legal_in_format(document: RetrievedDocument, mtg_format: Format) -> bool:
    name = document.metadata.get("name") or document.title
    if _is_basic_land(name):
        return True

    legalities = document.metadata.get("legalities") or {}
    if mtg_format == Format.casual:
        return any(value == "legal" for value in legalities.values())

    legality = legalities.get(mtg_format.value)
    return legality == "legal"


def _requested_color_set(request: DeckRequest) -> set[str]:
    return {color.upper() for color in request.colors if color.upper() in COLOR_TO_BASIC_LAND}


def _matches_requested_colors(document: RetrievedDocument, requested_colors: set[str]) -> bool:
    if not requested_colors:
        return True

    name = document.metadata.get("name") or document.title
    if _is_basic_land(name):
        return True

    identity = {str(color).upper() for color in document.metadata.get("color_identity") or []}
    colors = {str(color).upper() for color in document.metadata.get("colors") or []}
    card_colors = identity or colors
    return card_colors.issubset(requested_colors)


def _is_playable_card_document(
    document: RetrievedDocument,
    mtg_format: Format,
    requested_colors: set[str] | None = None,
) -> bool:
    return (
        document.source in LIVE_CARD_SOURCES
        and not _is_non_deck_game_object(document)
        and _is_legal_in_format(document, mtg_format)
        and _matches_requested_colors(document, requested_colors or set())
    )


def _is_candidate_nonland_document(
    document: RetrievedDocument,
    mtg_format: Format,
    requested_colors: set[str] | None = None,
) -> bool:
    return _is_playable_card_document(document, mtg_format, requested_colors) and not _is_land_document(document)


def _cards_from_retrieved_documents(
    documents: list[RetrievedDocument],
    mtg_format: Format,
    requested_colors: set[str] | None = None,
    max_cards: int = 9,
    request: DeckRequest | None = None,
) -> list[dict]:
    cards: list[dict] = []
    seen_names: set[str] = set()
    ranked_documents = rank_candidate_documents(documents, request, include_lands=False) if request is not None else documents

    for document in ranked_documents:
        if not _is_candidate_nonland_document(document, mtg_format, requested_colors):
            continue

        name = document.metadata.get("name") or document.title
        if not name or name in seen_names:
            continue

        seen_names.add(name)
        count = 1 if mtg_format == Format.commander or _is_basic_land(name) else 4
        card = {
            "name": name,
            "count": count,
            "role": document.metadata.get("type_line") or "retrieved from live Scryfall data",
            "mana_value": document.metadata.get("mana_value"),
        }
        if document.metadata.get("estimated_price_usd") is not None:
            card["estimated_price_usd"] = document.metadata.get("estimated_price_usd")
        cards.append(card)

        if len(cards) >= max_cards:
            break

    return cards


def _card_lookup(documents: list[RetrievedDocument]) -> dict[str, RetrievedDocument]:
    lookup: dict[str, RetrievedDocument] = {}
    for document in documents:
        name = document.metadata.get("name") or document.title
        if isinstance(name, str) and name:
            lookup[name] = document
    return lookup


def _enrich_card_from_document(card: dict, document: RetrievedDocument | None) -> dict:
    if document is None:
        return card
    enriched = dict(card)
    enriched.setdefault("role", document.metadata.get("type_line") or "agent-selected card")
    if document.metadata.get("type_line"):
        enriched["type_line"] = document.metadata.get("type_line")
    if document.metadata.get("mana_value") is not None:
        enriched["mana_value"] = document.metadata.get("mana_value")
    if document.metadata.get("estimated_price_usd") is not None:
        enriched["estimated_price_usd"] = document.metadata.get("estimated_price_usd")
    return enriched


def _merge_duplicate_cards(cards: list[dict], mtg_format: Format) -> list[dict]:
    merged: dict[str, dict] = {}
    order: list[str] = []
    for card in cards:
        name = str(card.get("name") or "").strip()
        if not name:
            continue
        if name not in merged:
            merged[name] = dict(card, name=name)
            order.append(name)
            continue
        existing = merged[name]
        existing["count"] = int(existing.get("count") or 0) + int(card.get("count") or 0)
        if not existing.get("role") and card.get("role"):
            existing["role"] = card.get("role")
        if existing.get("estimated_price_usd") is None and card.get("estimated_price_usd") is not None:
            existing["estimated_price_usd"] = card.get("estimated_price_usd")
        if existing.get("mana_value") is None and card.get("mana_value") is not None:
            existing["mana_value"] = card.get("mana_value")

    for card in merged.values():
        if not _is_basic_land(card["name"]):
            card["count"] = min(int(card.get("count") or 1), 1 if mtg_format == Format.commander else 4)
        else:
            card["count"] = int(card.get("count") or 1)
    return [merged[name] for name in order]


def _trim_to_max_deck_size(cards: list[dict], request: DeckRequest) -> None:
    target_size = _minimum_deck_size(request.format)
    while sum(card["count"] for card in cards) > target_size:
        nonland = next((card for card in reversed(cards) if not _is_land_card(card)), None)
        if nonland is None:
            nonland = cards[-1]
        nonland["count"] -= 1
        if nonland["count"] <= 0:
            cards.remove(nonland)


def _trim_nonlands_for_land_slots(cards: list[dict], request: DeckRequest) -> None:
    target_size = _minimum_deck_size(request.format)
    target_lands = _target_land_count(request, cards)
    max_nonlands = target_size - target_lands
    while sum(card["count"] for card in cards if not _is_land_card(card)) > max_nonlands:
        nonland = next((card for card in reversed(cards) if not _is_land_card(card)), None)
        if nonland is None:
            return
        nonland["count"] -= 1
        if nonland["count"] <= 0:
            cards.remove(nonland)


def _finalize_deck_cards(
    cards: list[dict],
    request: DeckRequest,
    documents: list[RetrievedDocument],
) -> list[dict]:
    lookup = _card_lookup(documents)
    enriched = [_enrich_card_from_document(card, lookup.get(str(card.get("name") or ""))) for card in cards]
    finalized = _merge_duplicate_cards(enriched, request.format)
    _trim_nonlands_for_land_slots(finalized, request)
    _shape_deck_size(finalized, request, documents)
    _trim_to_max_deck_size(finalized, request)
    return finalized


def _max_nonland_cards(mtg_format: Format) -> int:
    if mtg_format == Format.commander:
        return _minimum_deck_size(mtg_format) - 32
    return (_minimum_deck_size(mtg_format) - 18) // 4


def _minimum_deck_size(mtg_format: Format) -> int:
    return 100 if mtg_format == Format.commander else 60


def _target_land_count(request: DeckRequest, nonland_cards: list[dict]) -> int:
    if request.format == Format.commander:
        base = 37
        minimum = 32
        maximum = 42
    else:
        base = 24
        minimum = 18
        maximum = 28

    playstyle = request.playstyle.lower()
    if playstyle in {"aggro", "tempo"}:
        base -= 2
    elif playstyle in {"control", "ramp"}:
        base += 2

    mana_values = [
        int(card.get("mana_value") or 0)
        for card in nonland_cards
        if not _is_land_card(card) and card.get("mana_value") is not None
    ]
    if mana_values:
        average_mana_value = sum(mana_values) / len(mana_values)
        if average_mana_value <= 2:
            base -= 1
        elif average_mana_value >= 4:
            base += 1

    return max(minimum, min(maximum, base))


def _basic_lands_for_request(request: DeckRequest, documents: list[RetrievedDocument]) -> list[str]:
    requested_colors = [color.upper() for color in request.colors if color.upper() in COLOR_TO_BASIC_LAND]
    if requested_colors:
        return [COLOR_TO_BASIC_LAND[color] for color in requested_colors]

    inferred_colors: list[str] = []
    for document in documents:
        for color in document.metadata.get("color_identity") or document.metadata.get("colors") or []:
            color = color.upper()
            if color in COLOR_TO_BASIC_LAND and color not in inferred_colors:
                inferred_colors.append(color)

    if inferred_colors:
        return [COLOR_TO_BASIC_LAND[color] for color in inferred_colors]

    return ["Island", "Mountain"]


def _ensure_minimum_deck_size(
    cards: list[dict],
    request: DeckRequest,
    documents: list[RetrievedDocument],
) -> None:
    target_size = _minimum_deck_size(request.format)
    current_size = sum(card["count"] for card in cards)
    if current_size >= target_size:
        return

    lands = _basic_lands_for_request(request, documents)
    missing_count = target_size - current_size
    for index in range(missing_count):
        land_name = lands[index % len(lands)]
        existing_land = next((card for card in cards if card["name"] == land_name), None)
        if existing_land is None:
            cards.append({"name": land_name, "count": 1, "role": "mana source", "mana_value": 0})
        else:
            existing_land["count"] += 1


def _add_basic_lands(
    cards: list[dict],
    request: DeckRequest,
    documents: list[RetrievedDocument],
    count: int,
) -> None:
    lands = _basic_lands_for_request(request, documents)
    for index in range(count):
        land_name = lands[index % len(lands)]
        existing_land = next((card for card in cards if card["name"] == land_name), None)
        if existing_land is None:
            cards.append({"name": land_name, "count": 1, "role": "mana source", "mana_value": 0})
        else:
            existing_land["count"] += 1


def _land_price(document: RetrievedDocument) -> float:
    price = document.metadata.get("estimated_price_usd")
    if isinstance(price, int | float):
        return float(price)
    return 9999.0


def _land_quality_score(document: RetrievedDocument) -> float:
    text = " ".join(
        [
            document.title,
            str(document.metadata.get("type_line") or ""),
            document.content,
        ]
    ).lower()
    score = 0.0
    for term, weight in {
        "fetch": 5.0,
        "shock": 4.0,
        "surveil": 3.0,
        "triome": 3.0,
        "fast": 2.0,
        "pathway": 2.0,
        "pain": 1.5,
        "check": 1.0,
    }.items():
        if term in text:
            score += weight
    price = _land_price(document)
    if price < 9999.0:
        score += min(price / 10.0, 3.0)
    return score


def _recommended_land_documents(
    request: DeckRequest,
    documents: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    requested_colors = _requested_color_set(request)
    seen: set[str] = set()
    lands: list[RetrievedDocument] = []
    for document in documents:
        name = str(document.metadata.get("name") or document.title)
        if not name or name in seen or _is_basic_land(name):
            continue
        if not _is_land_document(document):
            continue
        if not _is_playable_card_document(document, request.format, requested_colors):
            continue
        seen.add(name)
        lands.append(document)

    if request.budget_usd is not None and request.budget_usd >= 300:
        return sorted(lands, key=lambda document: (-_land_quality_score(document), document.title))
    return sorted(lands, key=lambda document: (_land_price(document), document.title))


def _add_recommended_nonbasic_lands(
    cards: list[dict],
    request: DeckRequest,
    documents: list[RetrievedDocument],
    count: int,
) -> int:
    added = 0
    for document in _recommended_land_documents(request, documents):
        if added >= count:
            break
        name = str(document.metadata.get("name") or document.title)
        existing = next((card for card in cards if card["name"] == name), None)
        current_count = int(existing.get("count") or 0) if existing is not None else 0
        max_count = 1 if request.format == Format.commander else 4
        available_count = max_count - current_count
        if available_count <= 0:
            continue
        add_count = min(available_count, count - added)
        if existing is None:
            cards.append(
                {
                    "name": name,
                    "count": add_count,
                    "role": document.metadata.get("type_line") or "mana fixing",
                    "type_line": document.metadata.get("type_line"),
                    "mana_value": document.metadata.get("mana_value") or 0,
                    "estimated_price_usd": document.metadata.get("estimated_price_usd"),
                }
            )
        else:
            existing["count"] += add_count
            if existing.get("estimated_price_usd") is None and document.metadata.get("estimated_price_usd") is not None:
                existing["estimated_price_usd"] = document.metadata.get("estimated_price_usd")
        added += add_count
    return added


def _shape_deck_size(cards: list[dict], request: DeckRequest, documents: list[RetrievedDocument]) -> None:
    target_size = _minimum_deck_size(request.format)
    target_lands = _target_land_count(request, cards)
    current_size = sum(card["count"] for card in cards)
    current_lands = sum(card["count"] for card in cards if _is_land_card(card))

    if current_size >= target_size:
        return

    desired_lands_to_add = max(target_lands - current_lands, 0)
    lands_to_add = min(desired_lands_to_add, target_size - current_size)
    nonbasic_lands_added = _add_recommended_nonbasic_lands(cards, request, documents, lands_to_add)
    _add_basic_lands(cards, request, documents, lands_to_add - nonbasic_lands_added)

    _ensure_minimum_deck_size(cards, request, documents)


def _candidate_card_query(request: DeckRequest) -> str:
    playstyle_terms = PLAYSTYLE_QUERY_TERMS.get(request.playstyle.lower(), request.playstyle)
    requested_colors = [color.upper() for color in request.colors if color.upper() in COLOR_TO_NAME]
    color_terms = " ".join([*requested_colors, *(COLOR_TO_NAME[color] for color in requested_colors)])
    pair_name = COLOR_PAIR_NAMES.get(frozenset(requested_colors), "")
    must_include_terms = " ".join(request.must_include)
    return " ".join(
        [
            request.strategy,
            playstyle_terms,
            pair_name,
            color_terms,
            must_include_terms,
        ]
    ).strip()


def _context_terms(documents: list[RetrievedDocument], max_documents: int = 6) -> str:
    terms: list[str] = []
    for document in documents[:max_documents]:
        terms.append(document.title)
        for key in ("format", "section", "category"):
            value = document.metadata.get(key)
            if value:
                terms.append(str(value))
    return " ".join(terms)


def _card_query_from_context(
    request: DeckRequest,
    candidate_query: str,
    strategy_context: list[RetrievedDocument],
    rules_context: list[RetrievedDocument],
) -> str:
    return " ".join(
        term
        for term in [
            request.format.value,
            candidate_query,
            _context_terms(strategy_context, max_documents=8),
            _context_terms(rules_context, max_documents=3),
        ]
        if term
    ).strip()


def _format_search_term(mtg_format: Format) -> str:
    return "" if mtg_format == Format.casual else f"f:{mtg_format.value}"


def _color_identity_search_term(requested_colors: set[str]) -> str:
    if not requested_colors:
        return ""
    color_order = "WUBRG"
    return f"id<={''.join(color.lower() for color in color_order if color in requested_colors)}"


def _scryfall_text_terms(request: DeckRequest, strategy_context: list[RetrievedDocument]) -> list[str]:
    haystack = " ".join(
        [
            request.playstyle,
            request.strategy,
            " ".join(request.must_include),
            " ".join(document.title for document in strategy_context[:8]),
        ]
    ).lower()
    terms: list[str] = []
    for term in (
        "prowess",
        "surveil",
        "energy",
        "delirium",
        "reanimator",
        "affinity",
        "cascade",
        "storm",
        "burn",
        "tokens",
        "lifegain",
    ):
        if term in haystack:
            terms.append(term)
    return terms


def _scryfall_candidate_queries(
    request: DeckRequest,
    strategy_context: list[RetrievedDocument],
) -> list[str]:
    requested_colors = _requested_color_set(request)
    base_terms = [
        _format_search_term(request.format),
        _color_identity_search_term(requested_colors),
        "game:paper",
        "-is:digital",
        "-t:land",
    ]
    base = " ".join(term for term in base_terms if term)
    playstyle = request.playstyle.lower()
    queries = [
        f"{base} (t:creature or t:instant or t:sorcery)",
        f"{base} mv<=2 (t:creature or t:instant or t:sorcery)",
    ]

    if playstyle in {"aggro", "tempo"}:
        queries.append(f"{base} (o:haste or o:prowess or o:damage or o:draw)")
    elif playstyle == "control":
        queries.append(f"{base} (o:counter or o:destroy or o:exile or o:draw)")
    elif playstyle == "combo":
        queries.append(f"{base} (o:copy or o:search or o:untap or o:sacrifice)")
    elif playstyle == "ramp":
        queries.append(f"{base} (o:add or o:search) (t:creature or t:artifact or t:sorcery)")
    elif request.playstyle:
        queries.append(f"{base} {request.playstyle}")

    for term in _scryfall_text_terms(request, strategy_context):
        if term == "burn":
            queries.append(f"{base} (o:damage or o:prowess)")
        elif term == "tokens":
            queries.append(f"{base} o:create o:token")
        else:
            queries.append(f"{base} o:{term}")

    deduped: list[str] = []
    for query in queries:
        if query and query not in deduped:
            deduped.append(query)
    return deduped


async def _live_scryfall_card_context(
    request: DeckRequest,
    strategy_context: list[RetrievedDocument],
    max_cards: int,
    client: ScryfallClient | None = None,
) -> list[RetrievedDocument]:
    client = client or ScryfallClient()
    documents: list[RetrievedDocument] = []
    seen_names: set[str] = set()
    avoided_names = {name.lower() for name in request.avoid}

    async def add_card(card: dict[str, Any]) -> None:
        name = str(card.get("name") or "")
        if not name or name.lower() in avoided_names or name in seen_names:
            return
        document = scryfall_card_to_document(card)
        if not _is_candidate_nonland_document(document, request.format, _requested_color_set(request)):
            return
        seen_names.add(name)
        documents.append(document)

    for name in request.must_include:
        try:
            await add_card(await client.get_card_named(name))
        except httpx.HTTPError as exc:
            logger.warning(
                "Scryfall requested card lookup failed",
                extra=log_extra(name=name, error=str(exc)),
            )

    for query in _scryfall_candidate_queries(request, strategy_context):
        if len(documents) >= max_cards:
            break
        try:
            payload = await client.search_cards(query=query)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                continue
            logger.warning(
                "Scryfall candidate search failed",
                extra=log_extra(query=query, error=str(exc)),
            )
            continue
        except httpx.HTTPError as exc:
            logger.warning(
                "Scryfall candidate search failed",
                extra=log_extra(query=query, error=str(exc)),
            )
            continue

        for card in payload.get("data", []):
            await add_card(card)
            if len(documents) >= max_cards:
                break

    queries = _scryfall_candidate_queries(request, strategy_context)
    logger.info(
        "Live Scryfall card context assembled",
        extra=log_extra(card_context_count=len(documents), query_count=len(queries)),
    )
    return documents


def _strategy_query(request: DeckRequest, candidate_query: str) -> str:
    requested_colors = [color.upper() for color in request.colors if color.upper() in COLOR_TO_NAME]
    color_names = [COLOR_TO_NAME[color] for color in requested_colors]
    pair_name = COLOR_PAIR_NAMES.get(frozenset(requested_colors), "")
    return " ".join(
        term
        for term in [
            request.format.value,
            pair_name,
            " ".join(color_names),
            request.playstyle,
            request.strategy,
            candidate_query,
        ]
        if term
    ).strip()


def _general_strategy_query(request: DeckRequest) -> str:
    playstyle_terms = PLAYSTYLE_QUERY_TERMS.get(request.playstyle.lower(), request.playstyle)
    return " ".join(
        term
        for term in [
            request.playstyle,
            playstyle_terms,
            request.strategy,
            "deck building strategy mana curve interaction threats answers card advantage sideboard role balance",
        ]
        if term
    ).strip()


def _rules_prefixes_for_request(request: DeckRequest) -> list[str]:
    prefixes = ["100.2a", "100.4", "100.4a", "100.4b"]
    if request.format == Format.commander:
        prefixes.extend(["903.5a", "903.5b", "903.6"])
    return prefixes


def _rules_query_for_request(request: DeckRequest) -> str:
    intents = [
        "constructed deck minimum deck size exactly at least sixty cards",
        "maximum four copies of a card other than basic lands",
        f"{request.format.value} deck construction sideboard",
    ]
    if request.format == Format.commander:
        intents.extend(["commander singleton exactly one hundred cards", "commander color identity"])
    return " ".join(intents)


def _dedupe_documents(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    deduped: list[RetrievedDocument] = []
    seen: set[tuple[str, str, str]] = set()
    for document in documents:
        key = (
            document.source,
            str(document.metadata.get("rule_number") or document.metadata.get("name") or document.title),
            document.content[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(document)
    return deduped


def _is_strategy_document_for_format(document: RetrievedDocument, mtg_format: Format) -> bool:
    document_format = str(document.metadata.get("format") or "").lower()
    if not document_format:
        return True
    if mtg_format == Format.casual:
        return True
    return document_format == mtg_format.value


def _filter_strategy_documents(
    documents: list[RetrievedDocument],
    mtg_format: Format,
) -> list[RetrievedDocument]:
    return [document for document in documents if _is_strategy_document_for_format(document, mtg_format)]


def _response_context_documents(
    card_context: list[RetrievedDocument],
    rules_context: list[RetrievedDocument],
    strategy_context: list[RetrievedDocument],
    mtg_format: Format,
    requested_colors: set[str],
) -> list[RetrievedDocument]:
    meta_context = [item for item in strategy_context if item.source == "mtgdecks_meta_decks"]
    non_meta_strategy = [item for item in strategy_context if item.source != "mtgdecks_meta_decks"]
    return [
        *non_meta_strategy[:6],
        *meta_context[:6],
        *rules_context[:4],
        *[
            item
            for item in card_context
            if _is_candidate_nonland_document(item, mtg_format, requested_colors)
        ][:8],
    ]


def _coerce_model_cards(model_cards: list[dict], allowed_names: set[str]) -> list[dict]:
    cards: list[dict] = []
    for card in model_cards:
        name = str(card.get("name", "")).strip()
        count = int(card.get("count", 0) or 0)
        if not name or count <= 0 or name not in allowed_names:
            continue
        cards.append(
            {
                "name": name,
                "count": count,
                "role": str(card.get("role", "")).strip(),
                "mana_value": card.get("mana_value"),
                "estimated_price_usd": card.get("estimated_price_usd"),
            }
        )
    return cards


def _selected_cards_from_agent_result(
    agent_result: dict,
    request: DeckRequest,
    card_documents: list[RetrievedDocument],
) -> list[dict]:
    lookup = _card_lookup(card_documents)
    for payload in agent_result.get("tool_card_payloads", []):
        if not isinstance(payload, dict):
            continue
        metadata = payload.get("metadata") or {}
        name = metadata.get("name") or payload.get("title")
        if not isinstance(name, str) or not name:
            continue
        lookup[name] = RetrievedDocument(
            title=name,
            content=str(payload.get("content") or ""),
            source=str(payload.get("source") or "scryfall_live"),
            metadata=metadata,
        )
    cards: list[dict] = []
    for card in agent_result.get("selected_cards", []):
        if not isinstance(card, dict):
            continue
        name = str(card.get("name") or "").strip()
        if not name:
            continue
        document = lookup.get(name)
        requested_count = int(card.get("count") or (1 if request.format == Format.commander else 4))
        count = 1 if request.format == Format.commander else max(1, min(requested_count, 4))
        cards.append(
            _enrich_card_from_document(
                {
                    "name": name,
                    "count": count,
                    "role": card.get("role", "agent-selected card"),
                },
                document,
            )
        )
    return cards


def _documents_from_agent_card_payloads(agent_result: dict) -> list[RetrievedDocument]:
    documents: list[RetrievedDocument] = []
    for payload in agent_result.get("tool_card_payloads", []):
        if not isinstance(payload, dict):
            continue
        metadata = payload.get("metadata") or {}
        name = metadata.get("name") or payload.get("title")
        if not isinstance(name, str) or not name:
            continue
        documents.append(
            RetrievedDocument(
                title=name,
                content=str(payload.get("content") or ""),
                source=str(payload.get("source") or "scryfall_live"),
                metadata=metadata,
            )
        )
    return documents


def _documents_from_agent_context_payloads(agent_result: dict) -> list[RetrievedDocument]:
    documents: list[RetrievedDocument] = []
    for payload in agent_result.get("tool_context_payloads", []):
        if not isinstance(payload, dict):
            continue
        title = payload.get("title")
        source = payload.get("source")
        if not isinstance(title, str) or not isinstance(source, str):
            continue
        documents.append(
            RetrievedDocument(
                title=title,
                content=str(payload.get("content") or ""),
                source=source,
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                score=payload.get("score") if isinstance(payload.get("score"), int | float) else None,
            )
        )
    return documents


async def _enrich_prices_with_scryfall(
    cards: list[dict],
    client: ScryfallClient,
    steps: list[dict[str, str]],
) -> list[dict]:
    enriched_cards = [dict(card) for card in cards]
    checked = 0
    priced = 0
    for card in enriched_cards:
        name = str(card.get("name") or "").strip()
        if not name or _is_basic_land(name):
            continue
        checked += 1
        try:
            document = scryfall_card_to_document(await client.get_card_named(name))
        except httpx.HTTPError as exc:
            steps.append({"label": "Scryfall price check failed", "detail": f"{name} -> {exc}"})
            continue
        price = document.metadata.get("estimated_price_usd")
        if price is not None:
            card["estimated_price_usd"] = price
            priced += 1
        if card.get("role") in {None, "", "agent-selected card"}:
            card["role"] = document.metadata.get("type_line") or card.get("role") or "agent-selected card"
        if card.get("mana_value") is None and document.metadata.get("mana_value") is not None:
            card["mana_value"] = document.metadata.get("mana_value")

    steps.append(
        {
            "label": "Scryfall price check",
            "detail": f"Checked {checked} nonbasic cards after validation; found prices for {priced}.",
        }
    )
    return enriched_cards


def _agent_steps_with_fallback(
    steps: list[dict[str, str]],
    reason: str,
) -> list[dict[str, str]]:
    return [
        *steps,
        {
            "label": "Fallback",
            "detail": reason,
        },
    ]


@router.post("/generate", response_model=DeckResponse)
async def generate_deck(request: DeckRequest, session: Session = Depends(get_session)) -> DeckResponse:
    settings = get_settings()
    retriever = RagRetriever(session)
    logger.info(
        "Deck generation started",
        extra=log_extra(
            format=request.format.value,
            playstyle=request.playstyle,
            color_count=len(request.colors),
            must_include_count=len(request.must_include),
            openai_enabled=settings.openai_enabled,
        ),
    )
    requested_colors = _requested_color_set(request)
    candidate_query = _candidate_card_query(request)
    rules_context = retriever.search_text(
        query=_rules_query_for_request(request),
        limit=4,
        source="mtg_comprehensive_rules",
    )
    rules_context.extend(
        retriever.search_by_metadata_prefix(
            source="mtg_comprehensive_rules",
            metadata_key="rule_number",
            prefixes=_rules_prefixes_for_request(request),
            limit=8,
        )
    )
    rules_context = _dedupe_documents(rules_context)
    strategy_context = retriever.search_text(
        query=_strategy_query(request, candidate_query) or "deck building strategy",
        limit=60,
        source="mtgdecks_articles",
        metadata_filters={"format": [request.format.value, ""]} if request.format != Format.casual else None,
    )
    strategy_context = _filter_strategy_documents(strategy_context, request.format)[:18]
    general_strategy_query = _general_strategy_query(request)
    strategy_context = _dedupe_documents(
        [
            *strategy_context,
            *retriever.search_text(
                query=_strategy_query(request, candidate_query) or f"{request.format.value} metagame decks",
                limit=30,
                source="mtgdecks_meta_decks",
                metadata_filters={"format": [request.format.value]} if request.format != Format.casual else None,
            ),
            *retriever.search_text(
                query=general_strategy_query,
                limit=30,
                source="mtgdecks_articles",
                metadata_filters={"format": [""]},
            ),
            *retriever.search_text(
                query=general_strategy_query,
                limit=10,
                source="foundational_strategy",
            ),
        ]
    )[:30]
    card_query = _card_query_from_context(request, candidate_query, strategy_context, rules_context)
    max_nonland_cards = _max_nonland_cards(request.format)
    card_context = []
    if not settings.openai_agent_enabled:
        card_context = await _live_scryfall_card_context(
            request=request,
            strategy_context=strategy_context,
            max_cards=max_nonland_cards * 4,
        )
    logger.info(
        "Deck generation card context assembled",
        extra=log_extra(
            card_context_count=len(card_context),
            candidate_query_length=len(candidate_query),
            card_query_length=len(card_query),
        ),
    )

    cards = [
        {"name": name, "count": 1, "role": "requested card", "estimated_price_usd": None}
        for name in request.must_include
    ]

    retrieved_cards = _cards_from_retrieved_documents(
        card_context,
        request.format,
        requested_colors=requested_colors,
        max_cards=max_nonland_cards,
        request=request,
    )

    cards.extend(retrieved_cards)
    context = [*card_context, *rules_context]
    agent_steps: list[dict[str, str]] = [
        {
            "label": "Initial RAG context",
            "detail": (
                f"Retrieved {len(strategy_context)} strategy documents and "
                f"{len(rules_context)} rules documents before model planning."
            ),
        }
    ]
    if not settings.openai_agent_enabled:
        agent_steps.append(
            {
                "label": "Fallback candidate context",
                "detail": f"Fetched {len(card_context)} live Scryfall candidates before fallback drafting.",
            }
        )
    logger.info(
        "Deck generation context assembled",
        extra=log_extra(
            card_context_count=len(card_context),
            rules_context_count=len(rules_context),
            strategy_context_count=len(strategy_context),
        ),
    )

    if settings.openai_agent_enabled:
        try:
            agent_result = await OpenAIDeckAgent(
                settings=settings,
                tools=DeckAgentTools(retriever=retriever, scryfall=ScryfallClient()),
            ).generate(
                request=request,
                card_context=card_context,
                rules_context=rules_context,
                strategy_context=strategy_context,
                land_guidance={
                    "minimum": 32 if request.format == Format.commander else 18,
                    "default": _target_land_count(request, retrieved_cards),
                    "maximum": 42 if request.format == Format.commander else 28,
                },
            )
            agent_steps.extend(agent_result.get("agent_steps", []))
            card_context.extend(_documents_from_agent_card_payloads(agent_result))
            strategy_context = _dedupe_documents(
                [*strategy_context, *_documents_from_agent_context_payloads(agent_result)]
            )
            context = [*card_context, *rules_context]
            model_card_context = [
                item
                for item in card_context
                if _is_candidate_nonland_document(item, request.format, requested_colors)
            ]
            allowed_names = {
                *(item.metadata.get("name") or item.title for item in model_card_context),
                *agent_result.pop("tool_card_names", []),
                *COLOR_TO_BASIC_LAND.values(),
                "Wastes",
            }
            selected_cards = _selected_cards_from_agent_result(agent_result, request, model_card_context)
            model_cards = _coerce_model_cards(selected_cards, allowed_names)
            if model_cards:
                model_cards = _finalize_deck_cards(model_cards, request, context)
                validation = validate_deck(cards=model_cards, mtg_format=request.format)
                if not validation.is_valid:
                    raise ValueError(f"OpenAI agent deck failed validation: {validation.errors}")
                agent_steps.append(
                    {
                        "label": "Deck validation",
                        "detail": (
                            f"Deterministic skills finalized {sum(card['count'] for card in model_cards)} "
                            f"cards and validation returned {len(validation.errors)} errors."
                        ),
                    }
                )
                model_cards = await _enrich_prices_with_scryfall(
                    cards=model_cards,
                    client=ScryfallClient(),
                    steps=agent_steps,
                )
                mana_curve = calculate_mana_curve(model_cards)
                response_context = _response_context_documents(
                    card_context=card_context,
                    rules_context=rules_context,
                    strategy_context=strategy_context,
                    mtg_format=request.format,
                    requested_colors=requested_colors,
                )
                return DeckResponse(
                    title=agent_result.get("title") or f"{request.format.value.title()} OpenAI Agent Draft",
                    format=request.format,
                    cards=model_cards,
                    explanation=agent_result.get("explanation")
                    or "Selected by the OpenAI agent using RAG strategy, rules, and live Scryfall tools.",
                    mana_curve=mana_curve,
                    validation=validation,
                    retrieved_context=[f"{item.title} ({item.source})" for item in response_context],
                    agent_steps=agent_steps,
                    generation_mode=f"openai-agent:{settings.openai_model}",
                )
            raise ValueError("OpenAI agent returned no usable selected cards.")
        except Exception as exc:
            logger.warning(
                "OpenAI agent generation failed; falling back",
                extra=log_extra(model=settings.openai_model, error=str(exc)),
            )
            agent_steps = _agent_steps_with_fallback(
                agent_steps,
                f"OpenAI agent failed, so fallback generation continued: {exc}",
            )
            if not card_context:
                card_context = await _live_scryfall_card_context(
                    request=request,
                    strategy_context=strategy_context,
                    max_cards=max_nonland_cards * 4,
                )
                retrieved_cards = _cards_from_retrieved_documents(
                    card_context,
                    request.format,
                    requested_colors=requested_colors,
                    max_cards=max_nonland_cards,
                    request=request,
                )
                cards.extend(retrieved_cards)
                context = [*card_context, *rules_context]
    elif settings.ollama_enabled:
        try:
            agent_result = await OllamaDeckAgent(
                settings=settings,
                tools=DeckAgentTools(retriever=retriever, scryfall=ScryfallClient()),
            ).generate(
                request=request,
                card_context=card_context,
                rules_context=rules_context,
                strategy_context=strategy_context,
                land_guidance={
                    "minimum": 32 if request.format == Format.commander else 18,
                    "default": _target_land_count(request, retrieved_cards),
                    "maximum": 42 if request.format == Format.commander else 28,
                },
            )
            agent_steps.extend(agent_result.get("agent_steps", []))
            card_context.extend(_documents_from_agent_card_payloads(agent_result))
            strategy_context = _dedupe_documents(
                [*strategy_context, *_documents_from_agent_context_payloads(agent_result)]
            )
            context = [*card_context, *rules_context]
            model_card_context = [
                item
                for item in card_context
                if _is_candidate_nonland_document(item, request.format, requested_colors)
            ]
            allowed_names = {
                *(item.metadata.get("name") or item.title for item in model_card_context),
                *agent_result.pop("tool_card_names", []),
                *COLOR_TO_BASIC_LAND.values(),
                "Wastes",
            }
            selected_cards = _selected_cards_from_agent_result(agent_result, request, model_card_context)
            model_cards = _coerce_model_cards(selected_cards, allowed_names)
            if model_cards:
                model_cards = _finalize_deck_cards(model_cards, request, context)
                validation = validate_deck(cards=model_cards, mtg_format=request.format)
                if not validation.is_valid:
                    raise ValueError(f"Ollama deck failed validation: {validation.errors}")
                agent_steps.append(
                    {
                        "label": "Deck validation",
                        "detail": (
                            f"Deterministic skills finalized {sum(card['count'] for card in model_cards)} "
                            f"cards and validation returned {len(validation.errors)} errors."
                        ),
                    }
                )
                model_cards = await _enrich_prices_with_scryfall(
                    cards=model_cards,
                    client=ScryfallClient(),
                    steps=agent_steps,
                )
                mana_curve = calculate_mana_curve(model_cards)
                response_context = _response_context_documents(
                    card_context=card_context,
                    rules_context=rules_context,
                    strategy_context=strategy_context,
                    mtg_format=request.format,
                    requested_colors=requested_colors,
                )
                return DeckResponse(
                    title=agent_result.get("title") or f"{request.format.value.title()} Agent Draft",
                    format=request.format,
                    cards=model_cards,
                    explanation=agent_result.get("explanation")
                    or "Selected by the local Ollama agent using RAG strategy, rules, and live Scryfall tools.",
                    mana_curve=mana_curve,
                    validation=validation,
                    retrieved_context=[f"{item.title} ({item.source})" for item in response_context],
                    agent_steps=agent_steps,
                    generation_mode=f"ollama:{settings.ollama_model}",
                )
        except Exception as exc:
            logger.warning(
                "Ollama agent generation failed; falling back",
                extra=log_extra(model=settings.ollama_model, error=str(exc)),
            )
            agent_steps = _agent_steps_with_fallback(
                agent_steps,
                f"Ollama agent failed, so deterministic assembly continued: {exc}",
            )
    else:
        agent_steps.append(
            {
                "label": "Agent skipped",
                "detail": (
                    f"AGENT_PROVIDER is {settings.agent_provider!r}. Set AGENT_PROVIDER=openai "
                    "and OPENAI_API_KEY to let the OpenAI model plan and call tools."
                ),
            }
        )

    if settings.openai_enabled:
        try:
            model_card_context = [
                item
                for item in card_context
                if _is_candidate_nonland_document(item, request.format, requested_colors)
            ]
            model_result = await OpenAIDeckBuilder(settings).generate(
                request=request,
                card_context=model_card_context,
                rules_context=rules_context,
                strategy_context=strategy_context,
                land_guidance={
                    "minimum": 32 if request.format == Format.commander else 18,
                    "default": _target_land_count(request, []),
                    "maximum": 42 if request.format == Format.commander else 28,
                },
            )
            allowed_names = {
                *(item.metadata.get("name") or item.title for item in model_card_context),
                *COLOR_TO_BASIC_LAND.values(),
                "Wastes",
            }
            model_cards = _coerce_model_cards(model_result.get("cards", []), allowed_names)
            if model_cards:
                model_cards = _finalize_deck_cards(model_cards, request, context)
                validation = validate_deck(cards=model_cards, mtg_format=request.format)
                if not validation.is_valid:
                    raise ValueError(f"OpenAI deck failed validation: {validation.errors}")
                agent_steps.append(
                    {
                        "label": "OpenAI drafting",
                        "detail": "Model received retrieved strategy/rules context and candidate card context.",
                    }
                )
                agent_steps.append(
                    {
                        "label": "Deck validation",
                        "detail": (
                            f"Deterministic skills finalized {sum(card['count'] for card in model_cards)} "
                            f"cards and validation returned {len(validation.errors)} errors."
                        ),
                    }
                )
                model_cards = await _enrich_prices_with_scryfall(
                    cards=model_cards,
                    client=ScryfallClient(),
                    steps=agent_steps,
                )
                mana_curve = calculate_mana_curve(model_cards)
                response_context = _response_context_documents(
                    card_context=card_context,
                    rules_context=rules_context,
                    strategy_context=strategy_context,
                    mtg_format=request.format,
                    requested_colors=requested_colors,
                )
                return DeckResponse(
                    title=model_result.get("title") or f"{request.format.value.title()} Deck Draft",
                    format=request.format,
                    cards=model_cards,
                    explanation=model_result.get("explanation")
                    or "Generated from retrieved cards, rules, and strategy context.",
                    mana_curve=mana_curve,
                    validation=validation,
                    retrieved_context=[f"{item.title} ({item.source})" for item in response_context],
                    agent_steps=agent_steps,
                    generation_mode=f"openai:{settings.openai_model}",
                )
        except Exception as exc:
            logger.warning(
                "OpenAI deck generation failed; falling back to deterministic draft",
                extra=log_extra(error=str(exc)),
            )
            agent_steps = _agent_steps_with_fallback(
                agent_steps,
                f"OpenAI generation failed, so deterministic assembly continued: {exc}",
            )

    if not cards:
        cards = [
            {"name": "Lightning Bolt", "count": 4, "role": "efficient interaction", "mana_value": 1},
            {"name": "Consider", "count": 4, "role": "card selection", "mana_value": 1},
            {"name": "Island", "count": 8, "role": "mana source"},
            {"name": "Mountain", "count": 8, "role": "mana source"},
        ]

    cards = _finalize_deck_cards(cards, request, context)

    validation = validate_deck(cards=cards, mtg_format=request.format)
    agent_steps.append(
        {
            "label": "Deck validation",
            "detail": (
                f"Deterministic skills finalized {sum(card['count'] for card in cards)} cards and "
                f"validation returned {len(validation.errors)} errors."
            ),
        }
    )
    cards = await _enrich_prices_with_scryfall(
        cards=cards,
        client=ScryfallClient(),
        steps=agent_steps,
    )
    mana_curve = calculate_mana_curve(cards)
    response_context = _response_context_documents(
        card_context=card_context,
        rules_context=rules_context,
        strategy_context=strategy_context,
        mtg_format=request.format,
        requested_colors=requested_colors,
    )

    response = DeckResponse(
        title=f"{request.format.value.title()} {request.playstyle.title() or 'Deck'} Draft",
        format=request.format,
        cards=cards,
        explanation=(
            "This draft is assembled from requested cards, live Scryfall search results, and "
            "local strategy/rules context. The placeholder list is only used when live card "
            "lookup is unavailable."
        ),
        mana_curve=mana_curve,
        validation=validation,
        retrieved_context=[f"{item.title} ({item.source})" for item in response_context],
        agent_steps=agent_steps,
        generation_mode="deterministic",
    )
    logger.info(
        "Deck generation completed",
        extra=log_extra(
            card_count=len(response.cards),
            total_cards=sum(card.count for card in response.cards),
            is_valid=response.validation.is_valid,
            warning_count=len(response.validation.warnings),
            error_count=len(response.validation.errors),
        ),
    )
    return response
