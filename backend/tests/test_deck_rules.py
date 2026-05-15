from app.models.deck import Format
from app.skills.deck_rules import validate_deck


def test_modern_four_copy_limit_is_enforced() -> None:
    result = validate_deck(
        cards=[{"name": "Lightning Bolt", "count": 5}],
        mtg_format=Format.modern,
    )

    assert not result.is_valid
    assert "Lightning Bolt exceeds the four-copy limit." in result.errors


def test_basic_lands_are_exempt_from_four_copy_limit() -> None:
    result = validate_deck(
        cards=[{"name": "Island", "count": 60}],
        mtg_format=Format.modern,
    )

    assert result.is_valid


def test_deck_size_above_builder_target_is_invalid() -> None:
    result = validate_deck(
        cards=[{"name": "Island", "count": 61}],
        mtg_format=Format.modern,
    )

    assert not result.is_valid
    assert "Deck has 61 cards; this builder currently returns 60-card maindecks." in result.errors
