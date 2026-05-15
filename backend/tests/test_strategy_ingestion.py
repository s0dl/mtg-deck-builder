import json
from pathlib import Path

from app.rag.ingestion import load_strategy_article_index_file


def test_load_strategy_article_index_file_builds_rag_documents(tmp_path: Path) -> None:
    article_file = tmp_path / "articles.json"
    article_file.write_text(
        json.dumps(
            [
                {
                    "source": "mtgdecks_articles",
                    "url": "https://mtgdecks.net/guides/example-mtg-1",
                    "title": "Much Abrew: Example Tempo",
                    "category": "much abrew about nothing",
                    "section": "guides",
                    "summary": "A tempo deck with cheap threats and interaction.",
                    "published_at": "May 1",
                    "author": "SaffronOlive",
                }
            ]
        ),
        encoding="utf-8",
    )

    documents = load_strategy_article_index_file(article_file)

    assert len(documents) == 1
    assert documents[0].source == "mtgdecks_articles"
    assert documents[0].source_id == "https://mtgdecks.net/guides/example-mtg-1:0"
    assert documents[0].metadata["chunk_index"] == 0
    assert documents[0].metadata["category"] == "much abrew about nothing"
    assert documents[0].metadata["section"] == "guides"
    assert "cheap threats" in documents[0].content
