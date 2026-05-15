from app.skills.mana_curve import calculate_mana_curve


def test_calculate_mana_curve_excludes_lands() -> None:
    curve = calculate_mana_curve(
        [
            {"name": "Lightning Bolt", "count": 4, "mana_value": 1, "type_line": "Instant"},
            {"name": "Ledger Shredder", "count": 2, "mana_value": 2, "type_line": "Creature - Bird Advisor"},
            {"name": "Steam Vents", "count": 4, "mana_value": 0, "type_line": "Land - Island Mountain"},
            {"name": "Island", "count": 8, "mana_value": 0, "role": "mana source"},
        ]
    )

    assert [(bucket.mana_value, bucket.count) for bucket in curve] == [(1, 4), (2, 2)]
