from collections.abc import Iterable


def chunk_text(text: str, max_words: int = 260, overlap_words: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    step = max_words - overlap_words
    while start < len(words):
        chunk = words[start : start + max_words]
        chunks.append(" ".join(chunk))
        start += step

    return chunks


def join_card_fields(card: dict) -> str:
    legal_formats = [
        mtg_format
        for mtg_format, status in (card.get("legalities") or {}).items()
        if status == "legal"
    ]
    fields: Iterable[str | None] = (
        _labeled("Name", card.get("name")),
        _labeled("Type", card.get("type_line")),
        _labeled("Mana value", card.get("cmc")),
        _labeled("Colors", _join_list(card.get("colors"))),
        _labeled("Color identity", _join_list(card.get("color_identity"))),
        _labeled("Keywords", _join_list(card.get("keywords"))),
        _labeled("Legal formats", _join_list(legal_formats)),
        _labeled("Price USD", _price_usd(card)),
        _labeled("Oracle text", card.get("oracle_text")),
    )
    return "\n".join(field for field in fields if field)


def _join_list(value: object) -> str:
    if not isinstance(value, list):
        return ""
    return ", ".join(str(item) for item in value if item)


def _labeled(label: str, value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return f"{label}: {text}"


def _price_usd(card: dict) -> str:
    prices = card.get("prices")
    if not isinstance(prices, dict):
        return ""
    value = prices.get("usd")
    if value not in (None, ""):
        return str(value)
    for key in ("usd_foil", "usd_etched"):
        value = prices.get(key)
        if value not in (None, ""):
            return str(value)
    return ""
