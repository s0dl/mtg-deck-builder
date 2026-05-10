def within_budget(card: dict, max_price_usd: float | None) -> bool:
    if max_price_usd is None:
        return True
    price = card.get("estimated_price_usd")
    return price is not None and float(price) <= max_price_usd


def matches_colors(card: dict, colors: set[str]) -> bool:
    if not colors:
        return True
    card_colors = set(card.get("colors", []))
    return card_colors.issubset(colors)


def matches_type(card: dict, allowed_types: set[str]) -> bool:
    if not allowed_types:
        return True
    type_line = card.get("type_line", "").lower()
    return any(card_type.lower() in type_line for card_type in allowed_types)
