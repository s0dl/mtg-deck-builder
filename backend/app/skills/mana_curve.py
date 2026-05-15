from collections import Counter

from app.models.deck import ManaCurveBucket


def _is_land(card: dict) -> bool:
    name = str(card.get("name") or "")
    role = str(card.get("role") or "")
    type_line = str(card.get("type_line") or "")
    return (
        name in {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
        or "Land" in type_line.split(" ")
        or role == "mana source"
    )


def calculate_mana_curve(cards: list[dict]) -> list[ManaCurveBucket]:
    counts: Counter[int] = Counter()
    for card in cards:
        if _is_land(card):
            continue
        mana_value = int(card.get("mana_value", 0) or 0)
        counts[mana_value] += int(card["count"])

    return [
        ManaCurveBucket(mana_value=mana_value, count=count)
        for mana_value, count in sorted(counts.items())
    ]
