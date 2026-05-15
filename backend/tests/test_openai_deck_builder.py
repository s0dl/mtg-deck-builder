from app.llm.deck_builder import _extract_json_response


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
