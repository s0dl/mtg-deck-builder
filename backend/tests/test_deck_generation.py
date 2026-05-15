from app.api.decks import (
    _card_query_from_context,
    _cards_from_retrieved_documents,
    _ensure_minimum_deck_size,
    _filter_strategy_documents,
    _finalize_deck_cards,
    _general_strategy_query,
    _response_context_documents,
    _selected_cards_from_agent_result,
    _shape_deck_size,
    _scryfall_candidate_queries,
    _strategy_query,
    _target_land_count,
)
from app.mcp.scryfall_client import scryfall_card_to_document
from app.models.deck import DeckRequest, Format
from app.rag.retriever import RetrievedDocument


def make_card_document(
    name: str,
    legalities: dict[str, str] | None = None,
    source: str = "scryfall_bulk",
    type_line: str = "Instant",
    colors: list[str] | None = None,
    color_identity: list[str] | None = None,
) -> RetrievedDocument:
    return RetrievedDocument(
        title=name,
        content=f"{name} card text",
        source=source,
        metadata={
            "name": name,
            "mana_value": 1,
            "type_line": type_line,
            "colors": colors if colors is not None else [],
            "color_identity": color_identity if color_identity is not None else colors or [],
            "legalities": legalities if legalities is not None else {"modern": "legal", "commander": "legal"},
        },
    )


def test_cards_from_retrieved_documents_uses_scryfall_bulk_metadata() -> None:
    cards = _cards_from_retrieved_documents(
        [make_card_document("Monastery Swiftspear")],
        mtg_format=Format.modern,
    )

    assert cards == [
        {
            "name": "Monastery Swiftspear",
            "count": 4,
            "role": "Instant",
            "mana_value": 1,
        }
    ]


def test_cards_from_retrieved_documents_accepts_live_scryfall_documents() -> None:
    cards = _cards_from_retrieved_documents(
        [
            make_card_document(
                "Slickshot Show-Off",
                source="scryfall_live",
                type_line="Creature - Bird Wizard",
                colors=["R"],
                color_identity=["R"],
            )
        ],
        mtg_format=Format.modern,
        requested_colors={"R"},
    )

    assert cards[0]["name"] == "Slickshot Show-Off"
    assert cards[0]["count"] == 4


def test_cards_from_retrieved_documents_filters_non_card_and_illegal_rows() -> None:
    cards = _cards_from_retrieved_documents(
        [
            make_card_document("Strategy Notes", source="seed"),
            make_card_document("Ponder", legalities={"modern": "not_legal"}),
            make_card_document("Soldier Token", type_line="Token Creature - Soldier"),
            make_card_document("Elspeth Emblem", type_line="Emblem"),
            make_card_document("Consider", legalities={"modern": "legal"}),
        ],
        mtg_format=Format.modern,
    )

    assert [card["name"] for card in cards] == ["Consider"]


def test_cards_from_retrieved_documents_uses_singletons_for_commander() -> None:
    cards = _cards_from_retrieved_documents(
        [make_card_document("Sol Ring")],
        mtg_format=Format.commander,
    )

    assert cards[0]["count"] == 1


def test_cards_from_retrieved_documents_filters_not_legal_cards_for_casual() -> None:
    cards = _cards_from_retrieved_documents(
        [
            make_card_document("Ancestral Hot Dog Minotaur", legalities={"modern": "not_legal"}),
            make_card_document("Llanowar Elves", legalities={"modern": "legal"}),
        ],
        mtg_format=Format.casual,
    )

    assert [card["name"] for card in cards] == ["Llanowar Elves"]


def test_cards_from_retrieved_documents_filters_requested_color_identity() -> None:
    cards = _cards_from_retrieved_documents(
        [
            make_card_document("Expressive Iteration", colors=["U", "R"], color_identity=["U", "R"]),
            make_card_document("Drown in the Loch", colors=["U", "B"], color_identity=["U", "B"]),
            make_card_document("Lightning Bolt", colors=["R"], color_identity=["R"]),
        ],
        mtg_format=Format.modern,
        requested_colors={"U", "R"},
    )

    assert [card["name"] for card in cards] == ["Expressive Iteration", "Lightning Bolt"]


def test_ensure_minimum_deck_size_adds_requested_color_basics() -> None:
    cards = [{"name": "Monastery Swiftspear", "count": 4, "role": "threat", "mana_value": 1}]
    request = DeckRequest(format=Format.modern, colors=["R"])

    _ensure_minimum_deck_size(cards, request, documents=[])

    assert sum(card["count"] for card in cards) == 60
    assert next(card for card in cards if card["name"] == "Mountain")["count"] == 56


def test_ensure_minimum_deck_size_uses_commander_target() -> None:
    cards = [{"name": "Sol Ring", "count": 1, "role": "ramp", "mana_value": 1}]
    request = DeckRequest(format=Format.commander, colors=["U"])

    _ensure_minimum_deck_size(cards, request, documents=[])

    assert sum(card["count"] for card in cards) == 100
    assert next(card for card in cards if card["name"] == "Island")["count"] == 99


def test_shape_deck_size_uses_normal_constructed_land_count_when_spells_are_available() -> None:
    cards = [
        {"name": f"Spell {index}", "count": 4, "role": "spell", "mana_value": 1}
        for index in range(10)
    ]
    request = DeckRequest(format=Format.modern, colors=["U", "R"])

    _shape_deck_size(cards, request, documents=[])

    assert sum(card["count"] for card in cards) == 60
    assert sum(card["count"] for card in cards if card["role"] == "mana source") == 20


def test_target_land_count_changes_with_playstyle_and_curve() -> None:
    low_curve_cards = [
        {"name": f"Spell {index}", "count": 4, "role": "spell", "mana_value": 1}
        for index in range(9)
    ]
    high_curve_cards = [
        {"name": f"Spell {index}", "count": 4, "role": "spell", "mana_value": 5}
        for index in range(9)
    ]

    assert _target_land_count(DeckRequest(format=Format.modern, playstyle="aggro"), low_curve_cards) == 21
    assert _target_land_count(DeckRequest(format=Format.modern, playstyle="control"), high_curve_cards) == 27


def test_finalize_deck_cards_trims_to_target_size_and_preserves_prices() -> None:
    cards = [
        {"name": f"Spell {index}", "count": 4, "role": "spell", "mana_value": 1}
        for index in range(12)
    ]
    document = make_card_document("Spell 0", source="scryfall_live")
    document.metadata["estimated_price_usd"] = 1.25
    request = DeckRequest(format=Format.modern, colors=["U"])

    finalized = _finalize_deck_cards(cards, request, documents=[document])

    assert sum(card["count"] for card in finalized) == 60
    assert next(card for card in finalized if card["name"] == "Spell 0")["estimated_price_usd"] == 1.25


def test_finalize_deck_cards_reserves_land_slots_when_model_selects_too_many_spells() -> None:
    cards = [
        {"name": f"Spell {index}", "count": 4, "role": "spell", "mana_value": 1}
        for index in range(18)
    ]
    request = DeckRequest(format=Format.modern, colors=["R"], playstyle="tempo")

    finalized = _finalize_deck_cards(cards, request, documents=[])

    assert sum(card["count"] for card in finalized) == 60
    assert sum(card["count"] for card in finalized if card["role"] == "mana source") > 0


def test_finalize_deck_cards_counts_nonbasic_lands_as_lands() -> None:
    cards = [
        {"name": f"Spell {index}", "count": 4, "role": "spell", "mana_value": 1}
        for index in range(9)
    ]
    cards.extend(
        [
            {"name": "Steam Vents", "count": 4, "role": "mana fixing", "mana_value": 0},
            {"name": "Spirebluff Canal", "count": 4, "role": "mana fixing", "mana_value": 0},
        ]
    )
    documents = [
        make_card_document("Steam Vents", type_line="Land - Island Mountain", colors=[], color_identity=["U", "R"]),
        make_card_document("Spirebluff Canal", type_line="Land", colors=[], color_identity=["U", "R"]),
    ]
    request = DeckRequest(format=Format.modern, colors=["U", "R"], playstyle="tempo")

    finalized = _finalize_deck_cards(cards, request, documents=documents)

    assert sum(card["count"] for card in finalized) == 60
    assert next(card for card in finalized if card["name"] == "Steam Vents")["count"] == 4
    assert next(card for card in finalized if card["name"] == "Spirebluff Canal")["count"] == 4
    assert sum(card["count"] for card in finalized if card["role"] == "mana source") == 16


def test_finalize_deck_cards_prefers_available_nonbasic_lands_before_basics() -> None:
    cards = [
        {"name": f"Spell {index}", "count": 4, "role": "spell", "mana_value": 1}
        for index in range(9)
    ]
    steam_vents = make_card_document("Steam Vents", type_line="Land - Island Mountain", colors=[], color_identity=["U", "R"])
    steam_vents.metadata["estimated_price_usd"] = 8.0
    shivan_reef = make_card_document("Shivan Reef", type_line="Land", colors=[], color_identity=["U", "R"])
    shivan_reef.metadata["estimated_price_usd"] = 0.75
    request = DeckRequest(format=Format.modern, colors=["U", "R"], playstyle="tempo", budget_usd=100)

    finalized = _finalize_deck_cards(cards, request, documents=[steam_vents, shivan_reef])

    assert sum(card["count"] for card in finalized) == 60
    assert next(card for card in finalized if card["name"] == "Shivan Reef")["count"] == 4
    assert next(card for card in finalized if card["name"] == "Steam Vents")["count"] == 4
    assert sum(card["count"] for card in finalized if card["role"] == "mana source") < 21


def test_selected_cards_from_agent_result_preserves_variable_counts() -> None:
    agent_result = {
        "selected_cards": [
            {"name": "Lightning Bolt", "count": 4, "role": "core removal"},
            {"name": "Spell Pierce", "count": 2, "role": "situational interaction"},
            {"name": "Otawara, Soaring City", "count": 1, "role": "legendary utility land"},
        ]
    }
    documents = [
        make_card_document("Lightning Bolt", colors=["R"], color_identity=["R"]),
        make_card_document("Spell Pierce", colors=["U"], color_identity=["U"]),
        make_card_document("Otawara, Soaring City", type_line="Legendary Land", colors=[], color_identity=["U"]),
    ]
    request = DeckRequest(format=Format.modern, colors=["U", "R"])

    cards = _selected_cards_from_agent_result(agent_result, request, documents)

    assert [(card["name"], card["count"]) for card in cards] == [
        ("Lightning Bolt", 4),
        ("Spell Pierce", 2),
        ("Otawara, Soaring City", 1),
    ]


def test_selected_cards_from_agent_result_clamps_constructed_counts() -> None:
    agent_result = {"selected_cards": [{"name": "Lightning Bolt", "count": 9, "role": "core removal"}]}
    request = DeckRequest(format=Format.modern, colors=["R"])

    cards = _selected_cards_from_agent_result(agent_result, request, [make_card_document("Lightning Bolt")])

    assert cards[0]["count"] == 4


def test_selected_cards_from_agent_result_forces_commander_singletons() -> None:
    agent_result = {"selected_cards": [{"name": "Sol Ring", "count": 4, "role": "ramp"}]}
    request = DeckRequest(format=Format.commander)

    cards = _selected_cards_from_agent_result(agent_result, request, [make_card_document("Sol Ring")])

    assert cards[0]["count"] == 1


def test_filter_strategy_documents_removes_other_formats() -> None:
    modern = RetrievedDocument(
        title="Modern Guide",
        content="Modern article",
        source="mtgdecks_articles",
        metadata={"format": "modern"},
    )
    standard = RetrievedDocument(
        title="Standard Guide",
        content="Standard article",
        source="mtgdecks_articles",
        metadata={"format": "standard"},
    )
    general = RetrievedDocument(
        title="General Aggro Theory",
        content="General article",
        source="mtgdecks_articles",
        metadata={},
    )

    assert _filter_strategy_documents([modern, standard, general], Format.modern) == [modern, general]


def test_strategy_query_includes_format_color_names_and_pair_name() -> None:
    request = DeckRequest(
        format=Format.modern,
        colors=["U", "R"],
        playstyle="tempo",
        strategy="cheap threats and interaction",
    )

    query = _strategy_query(request, candidate_query="")

    assert "modern" in query
    assert "izzet" in query
    assert "blue red" in query
    assert "tempo" in query


def test_general_strategy_query_includes_broad_deck_building_terms() -> None:
    request = DeckRequest(
        format=Format.modern,
        colors=["U", "R"],
        playstyle="tempo",
        strategy="cheap threats and interaction",
    )

    query = _general_strategy_query(request)

    assert "tempo" in query
    assert "mana curve" in query
    assert "card advantage" in query
    assert "role balance" in query


def test_card_query_from_context_uses_strategy_and_rules_titles() -> None:
    request = DeckRequest(format=Format.modern, colors=["U", "R"], playstyle="tempo")
    strategy_context = [
        RetrievedDocument(
            title="Modern Izzet Prowess Deck Guide",
            content="",
            source="mtgdecks_articles",
            metadata={"format": "modern", "section": "guides"},
        )
    ]
    rules_context = [
        RetrievedDocument(
            title="Rule 100.2",
            content="",
            source="mtg_comprehensive_rules",
            metadata={"rule_number": "100.2"},
        )
    ]

    query = _card_query_from_context(request, "cheap threats", strategy_context, rules_context)

    assert "modern" in query
    assert "cheap threats" in query
    assert "Modern Izzet Prowess Deck Guide" in query
    assert "Rule 100.2" in query


def test_scryfall_candidate_queries_use_format_color_and_strategy_terms() -> None:
    request = DeckRequest(
        format=Format.modern,
        colors=["U", "R"],
        playstyle="tempo",
        strategy="prowess threats",
    )

    queries = _scryfall_candidate_queries(request, strategy_context=[])

    assert any("f:modern" in query for query in queries)
    assert any("id<=ur" in query for query in queries)
    assert any("o:prowess" in query for query in queries)


def test_scryfall_card_to_document_maps_live_card_truth() -> None:
    document = scryfall_card_to_document(
        {
            "name": "Lightning Bolt",
            "cmc": 1.0,
            "type_line": "Instant",
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
            "colors": ["R"],
            "color_identity": ["R"],
            "legalities": {"modern": "legal"},
            "prices": {"usd": "0.99"},
        }
    )

    assert document.source == "scryfall_live"
    assert document.metadata["name"] == "Lightning Bolt"
    assert document.metadata["estimated_price_usd"] == 0.99


def test_response_context_documents_orders_strategy_rules_then_cards() -> None:
    strategy = RetrievedDocument(
        title="How To Be A Better Player",
        content="",
        source="mtgdecks_articles",
        metadata={},
    )
    rule = RetrievedDocument(
        title="Rule 100.2",
        content="",
        source="mtg_comprehensive_rules",
        metadata={"rule_number": "100.2"},
    )
    card = make_card_document("Lightning Bolt", colors=["R"], color_identity=["R"])

    documents = _response_context_documents(
        card_context=[card],
        rules_context=[rule],
        strategy_context=[strategy],
        mtg_format=Format.modern,
        requested_colors={"U", "R"},
    )

    assert [document.title for document in documents] == [
        "How To Be A Better Player",
        "Rule 100.2",
        "Lightning Bolt",
    ]
