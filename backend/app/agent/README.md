# Agent Layer

This directory contains model-backed deck-building agents and the constrained tools they are allowed to use.

## Current State

- `OpenAIDeckAgent` uses the OpenAI Agents SDK to plan RAG calls for strategy/meta/rules context, then live Scryfall searches from that retrieved context, then card selection from those live results.
- `OllamaDeckAgent` remains available for local experimentation.
- `DeckAgentTools` is a client facade over `DeckBuilderMcpServer`.
- Strategy, meta-deck, rules, card corpus, live Scryfall, lookup, and validation tools are registered in `app/mcp/server.py`.
- Agent prompts receive tool signatures from the MCP server registry.

The backend executes tools and final validation. The model chooses what to ask for, but it does not get arbitrary code execution.
Live Scryfall is used for agent card discovery and after validation to refresh prices and card facts.

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
