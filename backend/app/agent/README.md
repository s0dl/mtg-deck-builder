# Agent Layer

This directory contains model-backed deck-building agents and the constrained tools they are allowed to use.

## Current State

- `OpenAIDeckAgent` receives initial strategy/rules RAG context, plans extra RAG calls, then selects cards from card-corpus results.
- `OllamaDeckAgent` remains available for local experimentation.
- `DeckAgentTools.search_strategy()` queries strategy/meta RAG documents.
- `DeckAgentTools.search_meta_decks()` queries MTGDecks archetype/top-deck snapshots directly.
- `DeckAgentTools.search_rules()` queries rules RAG documents.
- `DeckAgentTools.search_card_corpus()` queries embedded Scryfall card corpus documents using vector and text retrieval.
- `DeckAgentTools.search_cards_scryfall()` and `lookup_card()` provide guarded live Scryfall access.
- `DeckAgentTools.validate_deck_cards()` runs deterministic backend validation.

The backend executes tools and final validation. The model chooses what to ask for, but it does not get arbitrary code execution.
Live Scryfall can be used for candidate discovery when the embedded corpus is too thin, and is also used after validation to refresh prices and card facts.

## OpenAI Setup

```bash
AGENT_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4-mini
```

## Optional Ollama Setup

```bash
ollama pull llama3.2
AGENT_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

Smaller local models can fail to produce stable structured output. Keep deterministic finalization in backend code for every provider.
