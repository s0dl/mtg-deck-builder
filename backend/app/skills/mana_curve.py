from collections import Counter

from app.models.deck import ManaCurveBucket


def calculate_mana_curve(cards: list[dict]) -> list[ManaCurveBucket]:
    counts: Counter[int] = Counter()
    for card in cards:
        mana_value = int(card.get("mana_value", 0) or 0)
        counts[mana_value] += int(card["count"])

    return [
        ManaCurveBucket(mana_value=mana_value, count=count)
        for mana_value, count in sorted(counts.items())
    ]
