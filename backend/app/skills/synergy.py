def satisfies_synergy_tags(card: dict, required_tags: set[str]) -> bool:
    if not required_tags:
        return True
    tags = set(card.get("synergy_tags", []))
    return required_tags.issubset(tags)
