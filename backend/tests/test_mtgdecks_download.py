from scripts.scrapers.download_mtgdecks_articles import listing_url, parse_article, parse_listing


def test_listing_url_uses_wonky_pagination_path_after_first_page() -> None:
    assert listing_url("guides", 1) == "https://mtgdecks.net/guides"
    assert listing_url("guides", 2) == "https://mtgdecks.net/articles/index/guides/page:2"
    assert listing_url("meta", 3) == "https://mtgdecks.net/articles/index/meta/page:3"


def test_parse_listing_extracts_first_page_article_links() -> None:
    articles = parse_listing(
        """
        <html><body>
          <a href="/theory">Theory</a>
          <article>
            <a href="/theory/is-an-aggro-deck-really-an-aggro-deck-mtg-375">
              Is an Aggro Deck Really an Aggro Deck?
            </a>
          </article>
          <article>
            <a href="/guides/not-this-section-mtg-100">Ignore this</a>
          </article>
        </body></html>
        """,
        "https://mtgdecks.net/theory",
        "theory",
    )

    assert articles == [
        {
            "source": "mtgdecks_articles",
            "section": "theory",
            "url": "https://mtgdecks.net/theory/is-an-aggro-deck-really-an-aggro-deck-mtg-375",
            "title": "Is an Aggro Deck Really an Aggro Deck?",
        }
    ]


def test_parse_article_extracts_body_information() -> None:
    article = parse_article(
        """
        <html><body>
          <h1>Is an Aggro Deck Really an Aggro Deck?</h1>
          <p>MTGDecks</p>
          <p>2025-10-18 · 7 min read</p>
          <p>theory</p>
          <p>This article explains how pressure, curve, and interaction shape aggro strategy.</p>
          <h2>Deck construction</h2>
          <p>Low mana value threats need enough reach to close games after sweepers.</p>
          <p>Sign Up for MTGDecks newsletter</p>
        </body></html>
        """,
        "https://mtgdecks.net/theory/is-an-aggro-deck-really-an-aggro-deck-mtg-375",
        "theory",
    )

    assert article["source"] == "mtgdecks_articles"
    assert article["section"] == "theory"
    assert article["title"] == "Is an Aggro Deck Really an Aggro Deck?"
    assert article["format"] == ""
    assert article["published_at"] == "2025-10-18"
    assert article["read_time"] == "7 min read"
    assert article["author"] == "MTGDecks"
    assert "Low mana value threats" in article["content"]


def test_parse_article_infers_format_from_title() -> None:
    article = parse_article(
        """
        <html><body>
          <h1>Cosmogoyf Combo in Modern: Deck Tech & Sideboard Guide</h1>
          <p>Lucas Giggs</p>
          <p>2026-05-08 · 11 min read</p>
          <p>guides</p>
          <p>This article explains a combo deck and sideboard guide.</p>
        </body></html>
        """,
        "https://mtgdecks.net/guides/cosmogoyf-combo-in-modern-deck-tech-sideboard-guide-mtg-412",
        "guides",
    )

    assert article["format"] == "modern"
