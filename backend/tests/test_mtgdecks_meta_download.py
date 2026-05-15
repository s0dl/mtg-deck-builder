from scripts.download_mtgdecks_meta_decks import (
    format_url,
    parse_archetype_page,
    parse_decklist_page,
    parse_format_page,
)


def test_format_url_uses_format_endpoint() -> None:
    assert format_url("Modern") == "https://mtgdecks.net/Modern"


def test_parse_format_page_extracts_archetypes_and_percentage() -> None:
    archetypes = parse_format_page(
        """
        <html><body>
          <a href="/Standard/izzet-prowess">Izzet Prowess</a>
          <span>12.5%</span>
          <a href="/Standard/izzet-prowess-decklist-by-gnawe-2931400">Decklist</a>
          <a href="/Modern/not-this-format">Ignore</a>
        </body></html>
        """,
        "https://mtgdecks.net/Standard",
        "Standard",
        top=10,
    )

    assert archetypes == [
        {
            "format": "standard",
            "name": "Izzet Prowess",
            "url": "https://mtgdecks.net/Standard/izzet-prowess",
            "metagame_share": 12.5,
        }
    ]


def test_parse_format_page_skips_utility_links() -> None:
    archetypes = parse_format_page(
        """
        <html><body>
          <a href="/Standard/tournaments">Tournaments</a>
          <a href="/Standard/staples">Staples</a>
          <a href="/Standard/izzet-prowess">Izzet Prowess</a>
        </body></html>
        """,
        "https://mtgdecks.net/Standard",
        "Standard",
        top=10,
    )

    assert [item["name"] for item in archetypes] == ["Izzet Prowess"]


def test_parse_archetype_page_extracts_first_decklist() -> None:
    decklist_url = parse_archetype_page(
        """
        <html><body>
          <a href="/Standard/izzet-prowess-decklist-by-gnawe-2931400">Gnawe</a>
          <a href="/Standard/izzet-prowess-decklist-by-other-2931399">Other</a>
        </body></html>
        """,
        "https://mtgdecks.net/Standard/izzet-prowess",
        "Standard",
    )

    assert decklist_url == "https://mtgdecks.net/Standard/izzet-prowess-decklist-by-gnawe-2931400"


def test_parse_decklist_page_extracts_card_lines() -> None:
    decklist = parse_decklist_page(
        """
        <html><body>
          <h1>Izzet Prowess</h1>
          <p>4 Monastery Swiftspear</p>
          <p>4 Sleight of Hand</p>
          <p>2 Island</p>
          <p>Sideboard</p>
          <p>3 Negate</p>
        </body></html>
        """,
        "https://mtgdecks.net/Standard/izzet-prowess-decklist-by-gnawe-2931400",
    )

    assert {"count": 4, "name": "Monastery Swiftspear"} in decklist["cards"]
    assert {"count": 3, "name": "Negate"} in decklist["cards"]
