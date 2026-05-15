from app.agent.deck_builder import AGENT_PLAN_SCHEMA, _land_scryfall_queries
from app.llm.deck_builder import _extract_json_response
from app.models.deck import DeckRequest, Format


def test_extract_json_response_reads_responses_output_text() -> None:
    result = _extract_json_response(
        {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"title":"Deck","explanation":"Ok","cards":[]}',
                        }
                    ]
                }
            ]
        }
    )

    assert result["title"] == "Deck"


def test_agent_plan_schema_does_not_include_card_corpus_queries() -> None:
    assert "card_queries" not in AGENT_PLAN_SCHEMA["required"]
    assert "card_queries" not in AGENT_PLAN_SCHEMA["properties"]


def test_land_scryfall_queries_use_live_search_filters() -> None:
    queries = _land_scryfall_queries(
        DeckRequest(format=Format.modern, colors=["U", "R"], budget_usd=80)
    )

    assert queries
    assert all("f:modern" in query for query in queries)
    assert all("id<=ur" in query for query in queries)
    assert all("t:land" in query for query in queries)
    assert all("usd<5" in query for query in queries)
