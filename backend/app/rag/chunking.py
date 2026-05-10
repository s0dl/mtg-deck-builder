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
    fields: Iterable[str | None] = (
        card.get("name"),
        card.get("type_line"),
        card.get("oracle_text"),
        card.get("keywords") and ", ".join(card["keywords"]),
    )
    return "\n".join(field for field in fields if field)
