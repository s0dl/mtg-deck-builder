from app.models.deck import DeckValidation, Format

BASIC_LANDS = {
    "Plains",
    "Island",
    "Swamp",
    "Mountain",
    "Forest",
    "Wastes",
}


def validate_deck(cards: list[dict], mtg_format: Format) -> DeckValidation:
    total_cards = sum(card["count"] for card in cards)
    errors: list[str] = []
    warnings: list[str] = []

    minimum = 100 if mtg_format == Format.commander else 60
    if total_cards < minimum:
        warnings.append(f"Deck has {total_cards} cards; {mtg_format.value} usually needs {minimum}.")

    for card in cards:
        if card["name"] not in BASIC_LANDS and card["count"] > 4 and mtg_format != Format.commander:
            errors.append(f"{card['name']} exceeds the four-copy limit.")
        if mtg_format == Format.commander and card["name"] not in BASIC_LANDS and card["count"] > 1:
            errors.append(f"{card['name']} exceeds the commander singleton limit.")

    return DeckValidation(is_valid=not errors, errors=errors, warnings=warnings)
