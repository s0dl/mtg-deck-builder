# Agent Module

Path: `app/agent`

## Purpose

The agent module coordinates model-controlled deck construction without giving the model direct access to arbitrary code. The model can plan RAG and live-card searches; the backend executes only tools registered on the request-scoped MCP server.

## Main Files

- `context.py` - shared document-to-model payload formatting for agent prompts.
- `deck_builder.py` - OpenAI Agents SDK and Ollama agent implementations.
- `tools.py` - agent-facing client facade over `app.mcp.server.DeckBuilderMcpServer`.

## OpenAI Agent Flow

1. Load the `deck_builder_workflow` skill so every model phase sees the same ordered tool contract.
2. Receive the user request plus initial RAG strategy/rules context through an OpenAI Agents SDK agent with Pydantic structured output.
3. Ask the model for extra RAG strategy/meta/rules queries.
4. Execute those RAG tools through the MCP server and collect the returned documents.
5. Ask the model for live Scryfall searches using the retrieved documents.
6. Execute the live Scryfall searches through the MCP server.
7. Send tool results back for card selection.
8. Return selected cards, explanation, live card payloads, and agent steps.

The workflow skill constrains each phase to the right tool category:

- RAG planning: `search_strategy`, `search_meta_decks`, `search_rules`.
- Live card planning: `search_cards_scryfall`.
- Card selection: no tool calls; exact candidate names only.
- Validation/finalization: backend-owned `validate_deck_cards` and `lookup_card`.

The OpenAI agent intentionally receives rules and strategy context first. It then uses the retrieved RAG documents to drive guarded live Scryfall searches for candidate cards. Live Scryfall is also called by the backend after deterministic validation to refresh prices and card facts, through the same MCP server boundary.

## Tools

- `search_rag_text(query, limit, source, metadata_filters)` searches indexed RAG text.
- `search_rag_vector(query, limit, source)` searches indexed RAG vectors.
- `search_rag_metadata_prefixes(source, metadata_key, prefixes, limit)` searches indexed RAG metadata.
- `search_strategy(query, mtg_format, limit)` searches strategy and meta-deck RAG documents.
- `search_meta_decks(query, mtg_format, limit)` searches MTGDecks archetype/top-deck snapshots.
- `search_rules(intent, limit)` searches `mtg_comprehensive_rules`.
- `search_card_corpus(query, mtg_format, limit, request)` searches indexed Scryfall bulk card corpus.
- `search_cards_scryfall(query, limit)` searches live Scryfall for candidate cards.
- `lookup_card(name)` remains available for backend-controlled live checks.
- `validate_deck_cards(cards, mtg_format)` runs deterministic validation.

Tool responses are normalized as retrieved-document payloads so they can be displayed and reused by the API layer.
