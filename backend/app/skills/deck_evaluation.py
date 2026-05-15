from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.deck import DeckRequest, Format
from app.rag.retriever import RetrievedDocument

BASIC_LANDS = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
COLOR_NAMES = {
    "W": "white",
    "U": "blue",
    "B": "black",
    "R": "red",
    "G": "green",
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
STOP_WORDS = {
    "and",
    "are",
    "card",
    "cards",
    "deck",
    "for",
    "from",
    "into",
    "legal",
    "magic",
    "mana",
    "modern",
    "pioneer",
    "standard",
    "legacy",
    "vintage",
    "commander",
    "pauper",
    "casual",
    "the",
    "that",
    "this",
    "with",
    "you",
}


@dataclass(frozen=True)
class CandidateEvaluation:
    name: str
    score: float
    is_playable: bool
    role: str
    reasons: tuple[str, ...]


def rank_candidate_documents(
    documents: list[RetrievedDocument],
    request: DeckRequest,
    *,
    include_lands: bool = False,
) -> list[RetrievedDocument]:
    """Return playable card documents ordered by deck-building fit."""
    ranked = [
        (evaluate_candidate_document(document, request), index, document)
        for index, document in enumerate(documents)
    ]
    ranked = [
        (evaluation, index, document)
        for evaluation, index, document in ranked
        if evaluation.is_playable and (include_lands or evaluation.role != "land")
    ]
    ranked.sort(key=lambda item: (item[0].score, -item[1]), reverse=True)
    return [document for _, _, document in ranked]


def rank_candidate_cards(cards: list[dict[str, Any]], request: DeckRequest) -> list[dict[str, Any]]:
    """Return candidate card dictionaries ordered by deck-building fit."""
    ranked = [
        (evaluate_candidate_card(card, request), index, card)
        for index, card in enumerate(cards)
    ]
    ranked.sort(key=lambda item: (item[0].score, -item[1]), reverse=True)
    return [card for evaluation, _, card in ranked if evaluation.is_playable]


def evaluate_candidate_document(document: RetrievedDocument, request: DeckRequest) -> CandidateEvaluation:
    card = {
        "name": document.metadata.get("name") or document.title,
        "type_line": document.metadata.get("type_line"),
        "mana_value": document.metadata.get("mana_value"),
        "colors": document.metadata.get("colors"),
        "color_identity": document.metadata.get("color_identity"),
        "legalities": document.metadata.get("legalities"),
        "estimated_price_usd": document.metadata.get("estimated_price_usd"),
        "content": document.content,
        "source_score": document.score,
    }
    return evaluate_candidate_card(card, request)


def evaluate_candidate_card(card: dict[str, Any], request: DeckRequest) -> CandidateEvaluation:
    name = str(card.get("name") or "").strip()
    type_line = str(card.get("type_line") or "")
    role = _role_for_type_line(type_line)
    reasons: list[str] = []

    if not name:
        return CandidateEvaluation(name="", score=-1000.0, is_playable=False, role=role, reasons=("missing name",))
    if _has_non_deck_type_marker(type_line):
        return CandidateEvaluation(name=name, score=-1000.0, is_playable=False, role=role, reasons=("not a deck card",))
    if not _is_legal(card, request.format):
        return CandidateEvaluation(name=name, score=-1000.0, is_playable=False, role=role, reasons=("format legality",))
    if not _matches_color_identity(card, request):
        return CandidateEvaluation(name=name, score=-1000.0, is_playable=False, role=role, reasons=("color identity",))
    if _matches_avoid_terms(card, request):
        return CandidateEvaluation(name=name, score=-1000.0, is_playable=False, role=role, reasons=("avoid terms",))

    score = float(card.get("source_score") or 0.0)
    score += _curve_score(card, request)
    score += _role_score(role, request)
    score += _budget_score(card, request)
    synergy = _synergy_score(card, request)
    score += synergy
    if synergy > 0:
        reasons.append("matches strategy terms")
    if role in {"threat", "interaction", "card advantage", "ramp", "land"}:
        reasons.append(f"{role} role")
    if request.budget_usd is not None and _price(card) is not None:
        reasons.append("priced for budget")

    return CandidateEvaluation(name=name, score=score, is_playable=True, role=role, reasons=tuple(reasons))


def _is_legal(card: dict[str, Any], mtg_format: Format) -> bool:
    name = str(card.get("name") or "")
    if name in BASIC_LANDS:
        return True
    legalities = card.get("legalities") or {}
    if not isinstance(legalities, dict):
        return True
    if mtg_format == Format.casual:
        return any(value == "legal" for value in legalities.values())
    legality = legalities.get(mtg_format.value)
    return legality in {None, "legal"}


def _matches_color_identity(card: dict[str, Any], request: DeckRequest) -> bool:
    requested_colors = {color.upper() for color in request.colors if color.upper() in COLOR_NAMES}
    if not requested_colors:
        return True
    name = str(card.get("name") or "")
    if name in BASIC_LANDS:
        return True
    card_colors = {
        str(color).upper()
        for color in (card.get("color_identity") or card.get("colors") or [])
        if str(color).upper() in COLOR_NAMES
    }
    return card_colors.issubset(requested_colors)


def _matches_avoid_terms(card: dict[str, Any], request: DeckRequest) -> bool:
    haystack = _card_text(card)
    return any(str(term).strip().lower() in haystack for term in request.avoid if str(term).strip())


def _has_non_deck_type_marker(type_line: str) -> bool:
    return any(marker in type_line.split(" ") for marker in NON_DECK_TYPE_MARKERS)


def _role_for_type_line(type_line: str) -> str:
    lower = type_line.lower()
    if "land" in lower:
        return "land"
    if "creature" in lower or "planeswalker" in lower or "battle" in lower:
        return "threat"
    if "instant" in lower or "sorcery" in lower:
        return "interaction"
    if "artifact" in lower:
        return "ramp"
    if "enchantment" in lower:
        return "engine"
    return "support"


def _curve_score(card: dict[str, Any], request: DeckRequest) -> float:
    role = _role_for_type_line(str(card.get("type_line") or ""))
    if role == "land":
        return 3.0

    mana_value = _mana_value(card)
    if mana_value is None:
        return 0.0

    playstyle = request.playstyle.lower()
    if playstyle in {"aggro", "tempo"}:
        if mana_value <= 2:
            return 5.0
        if mana_value == 3:
            return 2.0
        return -2.0
    if playstyle == "control":
        if mana_value <= 2:
            return 2.0
        if 3 <= mana_value <= 5:
            return 3.0
        return -1.0
    if playstyle == "ramp":
        if mana_value <= 3 or role in {"ramp", "land"}:
            return 3.0
        return 1.0
    if playstyle == "combo":
        return 3.0 if mana_value <= 3 else 0.0
    return 2.0 if mana_value <= 3 else 0.0


def _role_score(role: str, request: DeckRequest) -> float:
    playstyle = request.playstyle.lower()
    preferred_roles = {
        "aggro": {"threat": 4.0, "interaction": 2.0, "land": 1.0},
        "tempo": {"threat": 3.0, "interaction": 4.0, "card advantage": 2.0, "land": 1.0},
        "control": {"interaction": 4.0, "card advantage": 3.0, "threat": 1.0, "land": 1.0},
        "combo": {"engine": 4.0, "support": 3.0, "card advantage": 3.0, "interaction": 1.0, "land": 1.0},
        "ramp": {"ramp": 4.0, "threat": 2.0, "land": 2.0},
        "midrange": {"threat": 3.0, "interaction": 3.0, "card advantage": 2.0, "land": 1.0},
    }
    return preferred_roles.get(playstyle, {}).get(role, 1.0)


def _budget_score(card: dict[str, Any], request: DeckRequest) -> float:
    if request.budget_usd is None:
        return 0.0
    price = _price(card)
    if price is None:
        return -0.5
    per_card_soft_cap = max(request.budget_usd / 20.0, 0.25)
    if price <= per_card_soft_cap:
        return 3.0
    if price <= per_card_soft_cap * 2:
        return 1.0
    return -min(6.0, price / max(per_card_soft_cap, 0.25))


def _synergy_score(card: dict[str, Any], request: DeckRequest) -> float:
    text = _card_text(card)
    terms = _request_terms(request)
    score = 0.0
    for term in terms:
        if term in text:
            score += 2.0
    for name in request.must_include:
        for term in _terms(str(name)):
            if term in text:
                score += 1.0
    return min(score, 12.0)


def _request_terms(request: DeckRequest) -> list[str]:
    raw = " ".join(
        [
            request.playstyle,
            request.strategy,
            " ".join(request.must_include),
            " ".join(COLOR_NAMES[color.upper()] for color in request.colors if color.upper() in COLOR_NAMES),
        ]
    )
    return _terms(raw)


def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw_term in text.lower().replace("-", " ").replace(":", " ").split():
        term = raw_term.strip("()[]{}.,;\"'")
        if len(term) <= 2 or term in STOP_WORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms


def _card_text(card: dict[str, Any]) -> str:
    parts = [
        str(card.get("name") or ""),
        str(card.get("type_line") or ""),
        str(card.get("oracle_text") or ""),
        str(card.get("content") or ""),
        " ".join(str(color) for color in card.get("colors") or []),
        " ".join(str(color) for color in card.get("color_identity") or []),
    ]
    return " ".join(parts).lower()


def _mana_value(card: dict[str, Any]) -> int | None:
    value = card.get("mana_value")
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _price(card: dict[str, Any]) -> float | None:
    price = card.get("estimated_price_usd")
    if price is None:
        return None
    try:
        return float(price)
    except (TypeError, ValueError):
        return None
